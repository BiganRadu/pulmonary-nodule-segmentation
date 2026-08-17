# Standard library
import os
import sys
import time
import random
import argparse
import multiprocessing

# Windows-specific environment configuration
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch_lib_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib_dir):
        if torch_lib_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = torch_lib_dir + ";" + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(torch_lib_dir)
            except Exception:
                pass

# PyTorch & MONAI (Must be imported before pandas/scipy on Windows to prevent WinError 1114 DLL conflicts)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import monai
from monai.networks.nets import UNet, AttentionUnet, SegResNet
from monai.losses import DiceFocalLoss, DiceCELoss, TverskyLoss
from monai.transforms import (
    Compose,
    Resized,
    RandRotated,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped
)

# 3rd party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import label, binary_erosion

# Default Configuration Constants
DEFAULT_MANIFEST = "preprocessed_data/dataset_manifest.csv"
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_MIN_LR = 1e-5
DEFAULT_VAL_MIN_SIZE = 0
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 8
DEFAULT_NEG_RATIO = 1.5
DEFAULT_LOSS = "dice_focal"
DEFAULT_MODEL_TYPE = "unet"
DEFAULT_SAVE_PATH = "models/unet_2.5d/unet_2.5d.pth"


def get_loss_function(loss_name):
    """
    Returns configured MONAI loss function instance based on loss_name.
    Supported: 'dice_focal', 'dice_ce', 'tversky'
    """
    name = loss_name.lower().replace("-", "_")
    if name in ["dice_ce", "dicece"]:
        return DiceCELoss(sigmoid=True, squared_pred=True)
    elif name in ["tversky"]:
        class TverskyCELoss(torch.nn.Module):
            def __init__(self, alpha=0.3, beta=0.7):
                super().__init__()
                self.tversky = TverskyLoss(sigmoid=True, alpha=alpha, beta=beta)
                self.bce = torch.nn.BCEWithLogitsLoss()
            def forward(self, input, target):
                return self.tversky(input, target) + self.bce(input, target)
        return TverskyCELoss(alpha=0.3, beta=0.7)
    elif name in ["dice_focal", "dicefocal"]:
        return DiceFocalLoss(sigmoid=True, squared_pred=True, gamma=2.0)
    else:
        raise ValueError(f"Unsupported loss function: '{loss_name}'. Choose from ['dice_focal', 'dice_ce', 'tversky']")


def get_model(model_type, in_channels=3):
    """
    Instantiates and returns (model, model_kwargs) based on model_type.
    Supported model types: 'unet', 'attention_unet', 'segresnet'
    """
    m_type = model_type.lower().replace("-", "_")
    if m_type in ["unet"]:
        model_kwargs = {
            "spatial_dims": 2,
            "in_channels": in_channels,
            "out_channels": 1,
            "channels": (16, 32, 64, 128, 256),
            "strides": (2, 2, 2, 2),
            "num_res_units": 2
        }
        model = UNet(**model_kwargs)
    elif m_type in ["attention_unet", "attentionunet", "att_unet", "attunet"]:
        model_kwargs = {
            "spatial_dims": 2,
            "in_channels": in_channels,
            "out_channels": 1,
            "channels": (16, 32, 64, 128, 256),
            "strides": (2, 2, 2, 2)
        }
        model = AttentionUnet(**model_kwargs)
    elif m_type in ["segresnet", "seg_resnet"]:
        model_kwargs = {
            "spatial_dims": 2,
            "in_channels": in_channels,
            "out_channels": 1,
            "init_filters": 16,
            "blocks_down": (1, 2, 2, 4),
            "blocks_up": (1, 1, 1)
        }
        model = SegResNet(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'. Choose from ['unet', 'attention_unet', 'segresnet']")

    return model, model_kwargs


def remove_small_objects(binary_mask, min_size):
    """
    Removes connected components in binary_mask that have fewer than min_size pixels.
    Eliminates small false positive noise predictions.
    """
    if min_size <= 0 or np.sum(binary_mask) == 0:
        return binary_mask
    labeled_mask, num_features = label(binary_mask)
    if num_features == 0:
        return binary_mask
    component_sizes = np.bincount(labeled_mask.ravel())
    too_small = component_sizes < min_size
    too_small[0] = False  # Ensure background is never removed
    cleaned_mask = binary_mask.copy()
    cleaned_mask[too_small[labeled_mask]] = 0
    return cleaned_mask


def compute_surface_distances(pred_binary, gt_binary):
    """
    Computes 95th Percentile Hausdorff Distance (HD95) and Average Surface Distance (ASD)
    between 2D/3D binary prediction and ground truth masks.
    Returns (np.nan, np.nan) if either mask is empty.
    """
    if np.sum(pred_binary) == 0 or np.sum(gt_binary) == 0:
        return np.nan, np.nan

    pred_border = pred_binary ^ binary_erosion(pred_binary)
    gt_border = gt_binary ^ binary_erosion(gt_binary)

    pred_pts = np.argwhere(pred_border)
    gt_pts = np.argwhere(gt_border)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan, np.nan

    d_p2g = np.min(np.linalg.norm(pred_pts[:, None, :] - gt_pts[None, :, :], axis=2), axis=1)
    d_g2p = np.min(np.linalg.norm(gt_pts[:, None, :] - pred_pts[None, :, :], axis=2), axis=1)

    all_dists = np.concatenate([d_p2g, d_g2p])
    hd95 = float(np.percentile(all_dists, 95))
    asd = float(np.mean(all_dists))
    return hd95, asd


def worker_init_fn(worker_id):
    """
    Worker initialization function for PyTorch DataLoaders (Cross-platform: Linux & Windows).
    Ensures each worker process gets a unique random seed for Python, NumPy, and PyTorch.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class LIDC25DDataset(Dataset):
    """
    PyTorch 2.5D Dataset for preprocessed LIDC-IDRI continuous slices.
    Loads 3 adjacent Z-slices [z-1, z, z+1] into a 3-channel input tensor (3, H, W).
    Target mask is 1-channel center slice mask [z] (1, H, W).
    Supports dynamic per-epoch negative slice resampling for the training split.
    """
    def __init__(self, manifest_csv, split="train", transform=None, neg_ratio=1.5, seed=42):
        df = pd.read_csv(manifest_csv)
        df["filepath"] = df["filepath"].str.replace("\\", "/", regex=False)
        self.split = split
        self.transform = transform
        self.neg_ratio = neg_ratio
        self.seed = seed
        self.full_split_df = df[df["split"] == split].reset_index(drop=True)

        # Build fast lookup map per patient across ALL slices for 3-slice context clamping
        self.patient_slices = {}
        for pid, group in df.groupby("patient_id"):
            sorted_group = group.sort_values("slice_idx")
            self.patient_slices[pid] = {
                "max_idx": sorted_group["slice_idx"].max(),
                "slice_map": dict(zip(sorted_group["slice_idx"], sorted_group["filepath"]))
            }

        if self.split == "train":
            self.resample_epoch(epoch=1)
        else:
            self.data_entries = self.full_split_df

    def resample_epoch(self, epoch=1):
        """
        Dynamically resamples negative background center slices per patient for the current epoch,
        retaining 100% of positive tumor slices and a fresh 1.5x subset of negative slices.
        Seed varies deterministically with epoch for full reproducibility across runs.
        """
        if self.split != "train":
            return

        epoch_seed = self.seed + int(epoch)
        rng = np.random.RandomState(epoch_seed)

        patient_sampled_list = []
        for pid, p_df in self.full_split_df.groupby("patient_id"):
            pos_df = p_df[p_df["has_tumor"] == 1]
            neg_df = p_df[p_df["has_tumor"] == 0]

            num_pos = len(pos_df)
            if num_pos > 0:
                target_neg_count = int(np.ceil(self.neg_ratio * num_pos))
            else:
                target_neg_count = min(len(neg_df), 20)

            if len(neg_df) > 0:
                sampled_neg_indices = rng.choice(len(neg_df), size=min(len(neg_df), target_neg_count), replace=False)
                sampled_neg_df = neg_df.iloc[sampled_neg_indices]
            else:
                sampled_neg_df = neg_df

            combined_p_df = pd.concat([pos_df, sampled_neg_df]).sort_values("slice_idx")
            patient_sampled_list.append(combined_p_df)

        self.data_entries = pd.concat(patient_sampled_list).sort_values(by=["patient_id", "slice_idx"]).reset_index(drop=True)

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        row = self.data_entries.iloc[idx]
        pid = str(row["patient_id"])
        z = int(row["slice_idx"])

        p_info = self.patient_slices[pid]
        max_z = p_info["max_idx"]
        slice_map = p_info["slice_map"]

        # Clamp boundary Z indices
        z_prev = max(0, z - 1)
        z_curr = z
        z_next = min(max_z, z + 1)

        path_prev = slice_map.get(z_prev, slice_map[z_curr])
        path_curr = slice_map[z_curr]
        path_next = slice_map.get(z_next, slice_map[z_curr])

        with np.load(path_prev) as npz_prev:
            img_prev = npz_prev["image"].astype(np.float32)
        with np.load(path_curr) as npz_curr:
            img_curr = npz_curr["image"].astype(np.float32)
            mask = npz_curr["mask"].astype(np.float32)
        with np.load(path_next) as npz_next:
            img_next = npz_next["image"].astype(np.float32)

        # Stack 3 1-channel arrays along channel dimension: shape (3, H, W)
        image_2_5d = np.concatenate([img_prev, img_curr, img_next], axis=0)

        image_tensor = torch.from_numpy(image_2_5d)
        mask_tensor = torch.from_numpy(mask)

        sample = {"image": image_tensor, "mask": mask_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["mask"], pid, z


def get_transforms(no_augmentations=False):
    """
    Returns MONAI transform pipelines with spatial resizing (256x256) and data augmentations for 2.5D inputs.
    If no_augmentations is True, train split uses deterministic resizing without random augmentations.
    """
    if no_augmentations:
        train_transforms = Compose([
            Resized(
                keys=["image", "mask"],
                spatial_size=(256, 256),
                mode=["bilinear", "nearest"]
            ),
            EnsureTyped(keys=["image", "mask"])
        ])
    else:
        train_transforms = Compose([
            Resized(
                keys=["image", "mask"],
                spatial_size=(256, 256),
                mode=["bilinear", "nearest"]
            ),
            RandRotated(
                keys=["image", "mask"],
                range_x=0.26,                  # ±15 degrees random rotation
                mode=["bilinear", "nearest"],
                prob=0.5
            ),
            RandFlipd(
                keys=["image", "mask"],
                spatial_axis=0,                # Horizontal flip
                prob=0.5
            ),
            RandFlipd(
                keys=["image", "mask"],
                spatial_axis=1,                # Vertical flip
                prob=0.5
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=0.2,
                std=0.03
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.3,
                gamma=(0.8, 1.2)
            ),
            EnsureTyped(keys=["image", "mask"])
        ])

    val_transforms = Compose([
        Resized(
            keys=["image", "mask"],
            spatial_size=(256, 256),
            mode=["bilinear", "nearest"]
        ),
        EnsureTyped(keys=["image", "mask"])
    ])

    return train_transforms, val_transforms


def plot_training_history(history, save_path="models/unet_2.5d/training_history.png"):
    """
    Plots training loss and validation metrics across 4 separate subplots.
    """
    epochs = history.get('epoch', [])
    if not epochs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Training Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='royalblue', linewidth=2)
    ax1.set_title('Training Loss (DiceFocalLoss)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # 2. Overlap & Composite Score
    ax2 = axes[0, 1]
    if 'val_dice' in history:
        ax2.plot(epochs, history['val_dice'], label='2D Dice (Tumor Only)', color='darkorange', linewidth=2)
    if 'val_iou' in history:
        ax2.plot(epochs, history['val_iou'], label='2D IoU', color='purple', linestyle='--', linewidth=1.5)
    if 'val_3d_dice' in history:
        ax2.plot(epochs, history['val_3d_dice'], label='3D Volumetric Dice', color='forestgreen', linewidth=2)
    if 'composite_score' in history:
        ax2.plot(epochs, history['composite_score'], label='Composite Score', color='crimson', linestyle=':', linewidth=2)
    ax2.set_title('Segmentation Overlap & Composite Score', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score [0-1]')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    # 3. Detection Rates
    ax3 = axes[1, 0]
    if 'val_prec' in history:
        ax3.plot(epochs, history['val_prec'], label='Precision', color='teal', linewidth=2)
    if 'val_sens' in history:
        ax3.plot(epochs, history['val_sens'], label='Sensitivity (Recall)', color='firebrick', linewidth=2)
    if 'val_spec' in history:
        ax3.plot(epochs, history['val_spec'], label='Specificity', color='darkgreen', linestyle='--', linewidth=1.5)
    ax3.set_title('Detection Rates (Precision, Sensitivity, Specificity)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Rate [0-1]')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()

    # 4. Boundary Error Distances
    ax4 = axes[1, 1]
    if 'val_hd95' in history:
        ax4.plot(epochs, history['val_hd95'], label='HD95 (mm)', color='darkred', linewidth=2)
    if 'val_asd' in history:
        ax4.plot(epochs, history['val_asd'], label='ASD (mm)', color='navy', linestyle='--', linewidth=1.5)
    ax4.set_title('Boundary Distance Errors (HD95 & ASD)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Distance Error (mm)')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_epoch(model, loader, optimizer, loss_fn, device, epoch, total_epochs, scaler=None):
    """
    Trains MONAI 2.5D UNet for 1 epoch using PyTorch AMP FP16 / FP32 precision.
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train 2.5D]", leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar, 1):
        images, masks = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val * images.size(0)
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    elapsed = time.time() - start_time
    return epoch_loss, elapsed


def validate_epoch(model, loader, device, val_min_size, use_amp=True):
    """
    Evaluates MONAI 2.5D UNet on validation set with macro-aggregated metrics and 3D volumetric reconstruction.
    """
    model.eval()

    patient_3d_preds = {}
    patient_3d_gts = {}

    sample_dices = []
    sample_ious = []
    sample_precisions = []
    sample_sensitivities = []
    sample_specificities = []
    sample_hd95s = []
    sample_asds = []
    failures = 0
    total_pos_samples = 0

    pbar = tqdm(loader, desc="[Val 2.5D]", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            images, masks, pids, s_idxs = batch[0], batch[1], batch[2], batch[3]
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(images)
            else:
                logits = model(images)

            preds_raw = (torch.sigmoid(logits) > 0.5).float()
            preds_np = preds_raw.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for b in range(preds_np.shape[0]):
                p_mask_bool = (preds_np[b, 0] > 0.5)
                g_mask_bool = (masks_np[b, 0] > 0.5)
                pid = pids[b]
                s_idx = int(s_idxs[b])

                if val_min_size > 0:
                    p_mask_bool = remove_small_objects(p_mask_bool, min_size=val_min_size)

                # Store lightweight boolean masks for 3D patient volume reconstruction
                if pid not in patient_3d_preds:
                    patient_3d_preds[pid] = {}
                    patient_3d_gts[pid] = {}
                patient_3d_preds[pid][s_idx] = p_mask_bool
                patient_3d_gts[pid][s_idx] = g_mask_bool

                g_sum = np.sum(g_mask_bool)
                if g_sum > 0:  # Macro-evaluation on positive tumor slices
                    total_pos_samples += 1
                    tp = np.sum(p_mask_bool & g_mask_bool)
                    fp = np.sum(p_mask_bool & ~g_mask_bool)
                    fn = np.sum(~p_mask_bool & g_mask_bool)
                    tn = np.sum(~p_mask_bool & ~g_mask_bool)

                    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
                    iou = tp / (tp + fp + fn + 1e-8)
                    prec = tp / (tp + fp + 1e-8)
                    sens = tp / (tp + fn + 1e-8)
                    spec = tn / (tn + fp + 1e-8)

                    sample_dices.append(dice)
                    sample_ious.append(iou)
                    sample_precisions.append(prec)
                    sample_sensitivities.append(sens)
                    sample_specificities.append(spec)

                    if dice < 0.1:  # Nodule detection failure threshold
                        failures += 1

                    # Surface distances (HD95 & ASD)
                    hd95, asd = compute_surface_distances(p_mask_bool, g_mask_bool)
                    if not np.isnan(hd95):
                        sample_hd95s.append(hd95)
                        sample_asds.append(asd)

    # Compute 3D Volumetric Dice on tumor-bearing patient scans
    patient_3d_dices = []
    for pid in patient_3d_preds:
        s_keys = sorted(patient_3d_preds[pid].keys())
        vol_pred = np.stack([patient_3d_preds[pid][k] for k in s_keys], axis=-1)
        vol_gt = np.stack([patient_3d_gts[pid][k] for k in s_keys], axis=-1)

        v_gt_sum = np.sum(vol_gt)
        if v_gt_sum > 0:
            inter = np.sum(vol_pred & vol_gt)
            v_pred_sum = np.sum(vol_pred)
            p_dice = (2.0 * inter) / (v_pred_sum + v_gt_sum + 1e-8)
            patient_3d_dices.append(p_dice)

    metrics = {
        "dice": float(np.mean(sample_dices)) if sample_dices else 0.0,
        "iou": float(np.mean(sample_ious)) if sample_ious else 0.0,
        "precision": float(np.mean(sample_precisions)) if sample_precisions else 0.0,
        "sensitivity": float(np.mean(sample_sensitivities)) if sample_sensitivities else 0.0,
        "specificity": float(np.mean(sample_specificities)) if sample_specificities else 0.0,
        "hd95": float(np.mean(sample_hd95s)) if sample_hd95s else 0.0,
        "asd": float(np.mean(sample_asds)) if sample_asds else 0.0,
        "failure_rate": float(failures / total_pos_samples) if total_pos_samples > 0 else 0.0,
        "val_3d_dice": float(np.mean(patient_3d_dices)) if patient_3d_dices else 0.0
    }

    metrics["composite_score"] = (metrics["dice"] + metrics["iou"] + metrics["sensitivity"]) / 3.0

    # Explicitly clear 3D volume dictionaries and trigger garbage collection
    del patient_3d_preds, patient_3d_gts
    import gc
    gc.collect()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MONAI 2.5D UNet Model on LIDC-IDRI Dataset")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help=f"Path to master manifest CSV (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Total target training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--min_lr", type=float, default=DEFAULT_MIN_LR, help=f"Minimum learning rate for scheduler (default: {DEFAULT_MIN_LR})")
    parser.add_argument("--val_min_size", type=int, default=DEFAULT_VAL_MIN_SIZE, help=f"Minimum connected component size (in pixels) during validation (default: {DEFAULT_VAL_MIN_SIZE})")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help=f"Weight decay for AdamW optimizer (default: {DEFAULT_WEIGHT_DECAY})")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help=f"DataLoader num_workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--neg_ratio", type=float, default=DEFAULT_NEG_RATIO, help=f"Ratio of negative slices to positive slices for dynamic epoch resampling (default: {DEFAULT_NEG_RATIO})")
    parser.add_argument("--loss", type=str, choices=["dice_focal", "dice_ce", "tversky"], default=DEFAULT_LOSS, help=f"Loss function to optimize: dice_focal, dice_ce, or tversky (default: {DEFAULT_LOSS})")
    parser.add_argument("--model_type", "--model", type=str, choices=["unet", "attention_unet", "segresnet"], default=DEFAULT_MODEL_TYPE, help=f"Model architecture to train: unet, attention_unet, or segresnet (default: {DEFAULT_MODEL_TYPE})")
    parser.add_argument("--no_transforms", "--no_aug", action="store_true", help="Disable random data augmentations during training (keeps 256x256 resizing only)")
    parser.add_argument("--save_path", "--model_path", type=str, default=DEFAULT_SAVE_PATH, help=f"Path to save best model checkpoint or output folder (default: {DEFAULT_SAVE_PATH})")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")

    args = parser.parse_args()

    # Automatically derive default save directory based on chosen model_type if default path was used
    if args.save_path == DEFAULT_SAVE_PATH and args.model_type != DEFAULT_MODEL_TYPE:
        args.save_path = f"models/{args.model_type}_2.5d/{args.model_type}_2.5d.pth"

    # Automatically derive all output artifact paths from --save_path directory
    raw_save_path = os.path.normpath(args.save_path)
    if raw_save_path.endswith(".pth"):
        save_dir = os.path.dirname(raw_save_path) or "."
        best_model_path = raw_save_path
    else:
        save_dir = raw_save_path
        best_model_path = os.path.join(save_dir, f"{args.model_type}_2.5d.pth")

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pth")
    results_log = os.path.join(save_dir, "train.txt")
    history_plot_path = os.path.join(save_dir, "training_history.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_transforms, val_transforms = get_transforms(no_augmentations=args.no_transforms)

    train_dataset = LIDC25DDataset(args.manifest, split="train", transform=train_transforms, neg_ratio=args.neg_ratio)
    val_dataset = LIDC25DDataset(args.manifest, split="val", transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=False,  # MUST be False so workers refresh and fetch newly resampled slices every epoch!
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None
    )

    model, model_kwargs = get_model(args.model_type, in_channels=3)
    model = model.to(device)

    loss_fn = get_loss_function(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True if device.type == 'cuda' else False)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    start_epoch = 1
    best_composite_score = 0.0
    history = {
        "epoch": [], "train_loss": [], "val_3d_dice": [], "val_dice": [],
        "val_iou": [], "val_prec": [], "val_sens": [], "val_spec": [],
        "val_hd95": [], "val_asd": [], "val_fail_rate": [], "composite_score": []
    }

    # Resume capability from checkpoint
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_composite_score = checkpoint.get("best_composite_score", 0.0)
        history = checkpoint.get("history", history)

        if "scaler_state_dict" in checkpoint and scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr

        remaining_epochs = max(1, args.epochs - start_epoch + 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=args.min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # Initialize results log file header if starting fresh
    if not args.resume or not os.path.exists(results_log):
        with open(results_log, "a") as f:
            f.write(f"\n--- Training Session Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write("Epoch | Loss | Dice | IoU | Precision | Sensitivity | Specificity | HD95(mm) | ASD(mm) | FailRate | 3DDice | CompositeScore\n")

    total_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_dataset, "resample_epoch"):
            train_dataset.resample_epoch(epoch=epoch)

        train_loss, train_time = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, args.epochs, scaler=scaler)
        val_metrics = validate_epoch(model, val_loader, device, args.val_min_size, use_amp=True)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_3d_dice"].append(val_metrics["val_3d_dice"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_prec"].append(val_metrics["precision"])
        history["val_sens"].append(val_metrics["sensitivity"])
        history["val_spec"].append(val_metrics["specificity"])
        history["val_hd95"].append(val_metrics["hd95"])
        history["val_asd"].append(val_metrics["asd"])
        history["val_fail_rate"].append(val_metrics["failure_rate"])
        history["composite_score"].append(val_metrics["composite_score"])

        is_best = val_metrics["composite_score"] > best_composite_score
        if is_best:
            best_composite_score = val_metrics["composite_score"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_metrics["dice"],
                "val_3d_dice": val_metrics["val_3d_dice"],
                "val_metrics": val_metrics,
                "model_kwargs": model_kwargs
            }, best_model_path)

        # Save checkpoint after EVERY epoch
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_composite_score": best_composite_score,
            "history": history,
            "model_kwargs": model_kwargs
        }
        torch.save(checkpoint_data, checkpoint_path)
        plot_training_history(history, save_path=history_plot_path)

        # Format clean metric log string
        log_line = (
            f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {train_loss:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | Prec: {val_metrics['precision']:.4f} | "
            f"Sens: {val_metrics['sensitivity']:.4f} | Spec: {val_metrics['specificity']:.4f} | "
            f"HD95: {val_metrics['hd95']:.2f}mm | ASD: {val_metrics['asd']:.2f}mm | "
            f"FailRate: {val_metrics['failure_rate']*100:.1f}% | 3DDice: {val_metrics['val_3d_dice']:.4f} | "
            f"Score: {val_metrics['composite_score']:.4f}"
        )

        # Print clean single-line metric status to console
        print(log_line, flush=True)

        # Write log entry to results.txt file
        with open(results_log, "a") as f:
            f.write(f"{epoch:02d} | {train_loss:.4f} | {val_metrics['dice']:.4f} | {val_metrics['iou']:.4f} | {val_metrics['precision']:.4f} | {val_metrics['sensitivity']:.4f} | {val_metrics['specificity']:.4f} | {val_metrics['hd95']:.2f} | {val_metrics['asd']:.2f} | {val_metrics['failure_rate']*100:.1f}% | {val_metrics['val_3d_dice']:.4f} | {val_metrics['composite_score']:.4f}\n")

    plot_training_history(history, save_path=history_plot_path)

if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
