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
from monai.networks.nets import UNet
from monai.networks.nets import AttentionUnet
from monai.losses import DiceFocalLoss
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import label, binary_erosion

# Default Configuration Constants
DEFAULT_MANIFEST = "preprocessed_data2/dataset_manifest.csv"
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_MIN_LR = 1e-5
DEFAULT_VAL_MIN_SIZE = 0
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 8
DEFAULT_SAVE_PATH = "models/unet/unet.pth"
DEFAULT_CHECKPOINT_PATH = "models/unet/latest_checkpoint.pth"
DEFAULT_RESULTS_LOG = "models/unet/train.txt"


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

    # Calculate surface point distances efficiently
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


class LIDC2DDataset(Dataset):
    """
    Standard PyTorch 2D Dataset for preprocessed LIDC-IDRI slices.
    Loads slice npz files directly from disk on-the-fly.
    """
    def __init__(self, manifest_csv, split="train", transform=None):
        df = pd.read_csv(manifest_csv)
        df["filepath"] = df["filepath"].str.replace("\\", "/", regex=False)
        self.data_entries = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        row = self.data_entries.iloc[idx]
        filepath = str(row["filepath"]).replace("\\", "/")
        with np.load(filepath) as npz_data:
            image = torch.from_numpy(npz_data["image"].astype(np.float32))
            mask = torch.from_numpy(npz_data["mask"].astype(np.float32))

        sample = {"image": image, "mask": mask}

        if self.transform:
            sample = self.transform(sample)

        pid = str(row["patient_id"])
        slice_idx = int(row["slice_idx"])
        return sample["image"], sample["mask"], pid, slice_idx


def get_transforms():
    """
    Returns MONAI transform pipelines with spatial resizing (256x256) and data augmentations.
    """
    train_transforms = Compose([
        Resized(
            keys=["image", "mask"],
            spatial_size=(256, 256),
            mode=["bilinear", "nearest"]
        ),
        RandRotated(
            keys=["image", "mask"],
            range_x=0.26,                  # ±15 degrees random rotation
            mode=["bilinear", "nearest"],  # Bilinear for image, exact nearest for mask
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


def plot_training_history(history, save_path="training_history.png"):
    """
    Plots training loss and validation metrics across 4 separate organized subplots:
    1. Training Loss
    2. Overlap & Composite Score (2D PosDice, 2D AllDice, 3D Dice, IoU, Composite Score)
    3. Detection & Classification Rates (Precision, Sensitivity, Specificity)
    4. Boundary Errors (HD95 & ASD in mm)
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
    if 'val_pos_dice' in history:
        ax2.plot(epochs, history['val_pos_dice'], label='2D Pos Dice (Tumor Only)', color='darkorange', linewidth=2)
    if 'val_all_dice' in history:
        ax2.plot(epochs, history['val_all_dice'], label='2D All Dice (All Slices)', color='deepskyblue', linestyle='--', linewidth=1.5)
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

    # 3. Detection & Classification Rates
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
    Trains MONAI UNet for 1 epoch using PyTorch AMP FP16 / FP32 precision.
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", leave=False, dynamic_ncols=True)
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


def validate_epoch(model, loader, device, use_amp=True, val_min_size=0):
    """
    Evaluates MONAI UNet on validation set with per-sample metrics:
    - 2D PosDice (Tumor Slices), 2D AllDice (All Slices), IoU, Precision, Sensitivity, Specificity, HD95, ASD, Failure Rate, 3D Volumetric Dice.
    """
    model.eval()

    patient_3d_preds = {}
    patient_3d_gts = {}

    sample_all_dices = []
    sample_pos_dices = []
    sample_ious = []
    sample_precisions = []
    sample_sensitivities = []
    sample_specificities = []
    sample_hd95s = []
    sample_asds = []
    failures = 0
    total_pos_samples = 0

    pbar = tqdm(loader, desc="[Val]", leave=False, dynamic_ncols=True)
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
                p_mask = preds_np[b, 0]
                g_mask = masks_np[b, 0]
                pid = pids[b]
                s_idx = int(s_idxs[b])

                if val_min_size > 0:
                    p_mask = remove_small_objects(p_mask, min_size=val_min_size)

                # Store for 3D patient volume reconstruction
                if pid not in patient_3d_preds:
                    patient_3d_preds[pid] = {}
                    patient_3d_gts[pid] = {}
                patient_3d_preds[pid][s_idx] = p_mask
                patient_3d_gts[pid][s_idx] = g_mask

                g_sum = np.sum(g_mask)
                p_sum = np.sum(p_mask)

                # 2D All Slices Dice computation
                if g_sum == 0 and p_sum == 0:
                    slice_all_dice = 1.0
                elif g_sum == 0 and p_sum > 0:
                    slice_all_dice = 0.0
                else:
                    tp_a = np.sum(p_mask * g_mask)
                    fp_a = np.sum(p_mask * (1.0 - g_mask))
                    fn_a = np.sum((1.0 - p_mask) * g_mask)
                    slice_all_dice = float((2.0 * tp_a) / (2.0 * tp_a + fp_a + fn_a + 1e-8))

                sample_all_dices.append(slice_all_dice)

                if g_sum > 0:  # Macro-evaluation on positive tumor slices
                    total_pos_samples += 1
                    tp = np.sum(p_mask * g_mask)
                    fp = np.sum(p_mask * (1.0 - g_mask))
                    fn = np.sum((1.0 - p_mask) * g_mask)
                    tn = np.sum((1.0 - p_mask) * (1.0 - g_mask))

                    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
                    iou = tp / (tp + fp + fn + 1e-8)
                    prec = tp / (tp + fp + 1e-8)
                    sens = tp / (tp + fn + 1e-8)
                    spec = tn / (tn + fp + 1e-8)

                    sample_pos_dices.append(dice)
                    sample_ious.append(iou)
                    sample_precisions.append(prec)
                    sample_sensitivities.append(sens)
                    sample_specificities.append(spec)

                    if dice < 0.1:  # Nodule detection failure threshold
                        failures += 1

                    # Surface distances (HD95 & ASD)
                    hd95, asd = compute_surface_distances(p_mask > 0, g_mask > 0)
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
            inter = np.sum(vol_pred * vol_gt)
            v_pred_sum = np.sum(vol_pred)
            p_dice = (2.0 * inter) / (v_pred_sum + v_gt_sum + 1e-8)
            patient_3d_dices.append(p_dice)

    pos_dice_mean = float(np.mean(sample_pos_dices)) if sample_pos_dices else 0.0
    all_dice_mean = float(np.mean(sample_all_dices)) if sample_all_dices else 0.0

    metrics = {
        "dice": pos_dice_mean,
        "dice_pos": pos_dice_mean,
        "dice_all": all_dice_mean,
        "iou": float(np.mean(sample_ious)) if sample_ious else 0.0,
        "precision": float(np.mean(sample_precisions)) if sample_precisions else 0.0,
        "sensitivity": float(np.mean(sample_sensitivities)) if sample_sensitivities else 0.0,
        "specificity": float(np.mean(sample_specificities)) if sample_specificities else 0.0,
        "hd95": float(np.mean(sample_hd95s)) if sample_hd95s else 0.0,
        "asd": float(np.mean(sample_asds)) if sample_asds else 0.0,
        "failure_rate": float(failures / total_pos_samples) if total_pos_samples > 0 else 0.0,
        "val_3d_dice": float(np.mean(patient_3d_dices)) if patient_3d_dices else 0.0
    }

    # Combined composite validation score for robust model selection (PosDice + IoU + Sensitivity)
    metrics["composite_score"] = (metrics["dice_pos"] + metrics["iou"] + metrics["sensitivity"]) / 3.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MONAI 2D UNet Model on LIDC-IDRI Dataset")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help=f"Path to master manifest CSV (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Total target training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--min_lr", type=float, default=DEFAULT_MIN_LR, help=f"Minimum learning rate for scheduler (default: {DEFAULT_MIN_LR})")
    parser.add_argument("--val_min_size", type=int, default=DEFAULT_VAL_MIN_SIZE, help=f"Minimum connected component size (in pixels) during validation (default: {DEFAULT_VAL_MIN_SIZE})")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help=f"Weight decay for AdamW optimizer (default: {DEFAULT_WEIGHT_DECAY})")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help=f"DataLoader num_workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH, help=f"Path to save best model checkpoint (default: {DEFAULT_SAVE_PATH})")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH, help=f"Path to save/load latest epoch checkpoint (default: {DEFAULT_CHECKPOINT_PATH})")
    parser.add_argument("--results_log", type=str, default=DEFAULT_RESULTS_LOG, help=f"Path to results log text file (default: {DEFAULT_RESULTS_LOG})")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_transforms, val_transforms = get_transforms()

    train_dataset = LIDC2DDataset(args.manifest, split="train", transform=train_transforms)
    val_dataset = LIDC2DDataset(args.manifest, split="val", transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
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

    unet_channels = (16, 32, 64, 128, 256)
    model_kwargs = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": unet_channels,
        "strides": (2, 2, 2, 2),
        "num_res_units": 2
    }

    model = UNet(**model_kwargs).to(device)

    loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True if device.type == 'cuda' else False)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    start_epoch = 1
    best_composite_score = 0.0
    history = {
        "epoch": [], "train_loss": [], "val_3d_dice": [], "val_pos_dice": [], "val_all_dice": [],
        "val_iou": [], "val_prec": [], "val_sens": [], "val_spec": [],
        "val_hd95": [], "val_asd": [], "val_fail_rate": [], "composite_score": []
    }

    # Resume capability from checkpoint
    if args.resume and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
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
    if not args.resume or not os.path.exists(args.results_log):
        with open(args.results_log, "a") as f:
            f.write(f"\n--- Training Session Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write("Epoch | Loss | PosDice | AllDice | IoU | Precision | Sensitivity | Specificity | HD95(mm) | ASD(mm) | FailRate | 3DDice | CompositeScore\n")

    total_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_time = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, args.epochs, scaler=scaler)
        val_metrics = validate_epoch(model, val_loader, device, use_amp=True, val_min_size=args.val_min_size)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_3d_dice"].append(val_metrics["val_3d_dice"])
        history["val_pos_dice"].append(val_metrics["dice_pos"])
        history["val_all_dice"].append(val_metrics["dice_all"])
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
            abs_save_path = os.path.abspath(args.save_path)
            os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_metrics["dice_pos"],
                "val_3d_dice": val_metrics["val_3d_dice"],
                "val_metrics": val_metrics,
                "model_kwargs": model_kwargs
            }, abs_save_path)

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
        torch.save(checkpoint_data, args.checkpoint_path)
        plot_training_history(history, save_path="training_history.png")

        # Format clean metric log string
        log_line = (
            f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {train_loss:.4f} | "
            f"PosDice: {val_metrics['dice_pos']:.4f} | AllDice: {val_metrics['dice_all']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | Prec: {val_metrics['precision']:.4f} | "
            f"Sens: {val_metrics['sensitivity']:.4f} | Spec: {val_metrics['specificity']:.4f} | "
            f"HD95: {val_metrics['hd95']:.2f}mm | ASD: {val_metrics['asd']:.2f}mm | "
            f"FailRate: {val_metrics['failure_rate']*100:.1f}% | 3DDice: {val_metrics['val_3d_dice']:.4f} | "
            f"Score: {val_metrics['composite_score']:.4f}"
        )

        # Print clean single-line metric status to console
        print(log_line, flush=True)

        # Write log entry to results.txt file
        with open(args.results_log, "a") as f:
            f.write(f"{epoch:02d} | {train_loss:.4f} | {val_metrics['dice_pos']:.4f} | {val_metrics['dice_all']:.4f} | {val_metrics['iou']:.4f} | {val_metrics['precision']:.4f} | {val_metrics['sensitivity']:.4f} | {val_metrics['specificity']:.4f} | {val_metrics['hd95']:.2f} | {val_metrics['asd']:.2f} | {val_metrics['failure_rate']*100:.1f}% | {val_metrics['val_3d_dice']:.4f} | {val_metrics['composite_score']:.4f}\n")

    plot_training_history(history, save_path="training_history.png")

if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
