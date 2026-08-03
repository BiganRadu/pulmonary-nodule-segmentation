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

# 3rd party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import label

# PyTorch & MONAI
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import monai
from monai.networks.nets import UNet, AttentionUnet
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    RandRotated,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped
)

# Default Configuration Constants
DEFAULT_MANIFEST = "preprocessed_data/dataset_manifest.csv"
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_MIN_LR = 1e-5
DEFAULT_VAL_MIN_SIZE = 30
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 8
DEFAULT_SAVE_PATH = "models/attention_unet/attention_unet.pth"
DEFAULT_CHECKPOINT_PATH = "latest_checkpoint.pth"

def remove_small_objects(binary_mask, min_size=30):
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

def worker_init_fn(worker_id):
    """
    Worker initialization function for PyTorch DataLoaders (Cross-platform: Linux & Windows).
    Ensures each worker process gets a unique random seed for Python, NumPy, and PyTorch,
    preventing duplicate random data augmentations across multi-process DataLoader workers.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class LIDC2DDataset(Dataset):
    """
    Standard PyTorch 2D Dataset for LIDC-IDRI slices.
    Loads slice npz files directly from disk on-the-fly with context management for file safety.
    """
    def __init__(self, manifest_csv, split="train", transform=None):
        df = pd.read_csv(manifest_csv)
        self.data_entries = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

        pos_count = int((self.data_entries['has_tumor'] == 1).sum())
        neg_count = int((self.data_entries['has_tumor'] == 0).sum())
        print(f"[{split.upper()} Set] Loaded {len(self.data_entries)} 2D slice pairs ({pos_count} Positive, {neg_count} Negative).")

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

        return sample["image"], sample["mask"]

def get_transforms():
    """
    Returns MONAI transform pipelines with data augmentations.
    """
    train_transforms = Compose([
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
        EnsureTyped(keys=["image", "mask"])
    ])

    return train_transforms, val_transforms

def plot_training_history(history, save_path="training_history.png"):
    """
    Plots and saves loss and validation Dice score curves over training epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.set_title('Training Loss (DiceCELoss)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(history['epoch'], history['val_dice'], label='Val Dice Score', color='green', linewidth=2)
    ax2.set_title('Validation Dice Coefficient', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Metric')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves plot to: {save_path}")

def train_epoch(model, loader, optimizer, loss_fn, device, epoch, total_epochs, scaler=None):
    """
    Trains MONAI UNet for 1 epoch using PyTorch AMP FP16 / FP32 precision with progress tracking.
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", leave=False, dynamic_ncols=True)
    for batch_idx, (images, masks) in enumerate(pbar, 1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
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
def validate_epoch(model, loader, dice_metric, device, use_amp=True, val_min_size=30):
    """
    Evaluates MONAI UNet on validation set with optional AMP FP16 precision.
    Applies small connected component post-processing filter (val_min_size px) to eliminate false positives.
    """
    model.eval()
    dice_metric.reset()

    pbar = tqdm(loader, desc="[Val]", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(images)
            else:
                logits = model(images)

            preds_raw = (torch.sigmoid(logits) > 0.5).float()

            if val_min_size > 0:
                preds_np = preds_raw.cpu().numpy()
                for b in range(preds_np.shape[0]):
                    preds_np[b, 0] = remove_small_objects(preds_np[b, 0], min_size=val_min_size)
                preds = torch.from_numpy(preds_np).to(device)
            else:
                preds = preds_raw

            dice_metric(y_pred=preds, y=masks)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice


def main():
    parser = argparse.ArgumentParser(description="Train 2D MONAI UNet Model on LIDC-IDRI Dataset (Accelerated AMP FP16 & Cross-Platform)")
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
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")

    args = parser.parse_args()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=======================================================")
    print(f"Using Compute Device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True
    print(f"DataLoader Multi-Worker Settings: num_workers={args.num_workers}, persistent_workers={args.num_workers > 0}, pin_memory={device.type == 'cuda'}")
    print(f"=======================================================\n")

    # Load Transforms & Datasets
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

    # Instantiate Wider MONAI 2D UNet Architecture
    unet_channels = (16, 32, 64, 128, 256)
    model_kwargs = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": unet_channels,
        "strides": (2, 2, 2, 2),
        #"num_res_units": 2
    }
    model = AttentionUnet(**model_kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 1
    best_val_dice = 0.0
    history = {"epoch": [], "train_loss": [], "val_dice": []}

    # Resume capability from checkpoint
    if args.resume:
        if os.path.exists(args.checkpoint_path):
            print(f"--> Found existing checkpoint '{args.checkpoint_path}'. Resuming training...")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            best_val_dice = checkpoint.get("best_val_dice", 0.0)
            history = checkpoint.get("history", history)

            if "scaler_state_dict" in checkpoint and scaler is not None and checkpoint.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            if start_epoch > args.epochs:
                print(f"--> Note: Checkpoint is at epoch {checkpoint['epoch']}, but target --epochs is set to {args.epochs}.")
                print(f"    Please pass a higher --epochs argument (e.g., --epochs {checkpoint['epoch'] + 30}) to train further.")

            # Set learning rate to args.lr for continued training
            old_lr = optimizer.param_groups[0]["lr"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
            print(f"--> Active Learning Rate set to: {args.lr:.2e} (was {old_lr:.2e}).")

            # Create smooth scheduler for remaining epochs
            remaining_epochs = max(1, args.epochs - start_epoch + 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=args.min_lr)
            print(f"--> Resuming from Epoch {start_epoch}. Continuing to target Epoch {args.epochs} ({remaining_epochs} epochs remaining). Previous Best Val Dice: {best_val_dice:.4f}\n")
        else:
            print(f"--> Note: --resume flag was set, but checkpoint file '{args.checkpoint_path}' does not exist yet. Starting training from scratch.\n")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)


    print(f"\nStarting Training: Wider MONAI 2D UNet (AMP FP16) | Channels: {unet_channels} ({num_params/1e6:.2f}M Params) | Target Epochs: {args.epochs} | Batch Size: {args.batch_size} | Workers: {args.num_workers} | Active LR: {optimizer.param_groups[0]['lr']:.2e}\n", flush=True)

    total_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_time = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, args.epochs, scaler=scaler)
        val_dice = validate_epoch(model, val_loader, dice_metric, device, use_amp=True, val_min_size=args.val_min_size)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            abs_save_path = os.path.abspath(args.save_path)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "model_kwargs": model_kwargs
            }, abs_save_path)
            best_str = f" -> [BEST MODEL SAVED to {abs_save_path}]"
        else:
            best_str = ""

        # Save checkpoint after EVERY epoch to allow resuming if PC crashes
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_dice": best_val_dice,
            "history": history,
            "model_kwargs": model_kwargs
        }
        torch.save(checkpoint_data, args.checkpoint_path)
        # Update training history plot after every epoch
        plot_training_history(history, save_path="training_history.png")

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.2e} | Time: {train_time:.1f}s{best_str}")

    total_elapsed = time.time() - total_start

    print(f"\n=======================================================")
    print(f"Training Complete in {total_elapsed / 60:.2f} minutes!")
    print(f"Best Validation Dice Score: {best_val_dice:.4f}")
    print(f"Saved Checkpoint to: {args.save_path}")
    print(f"=======================================================")

    plot_training_history(history, save_path="training_history.png")

if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
