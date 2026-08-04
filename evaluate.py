# Standard library
import os
import sys
import argparse
import multiprocessing

# Windows-specific environment configuration
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
from scipy.ndimage import label
from tqdm import tqdm

# PyTorch & MONAI
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.networks.nets import UNet

# Local imports
from train import LIDC2DDataset, get_transforms

# Default Configuration Constants
DEFAULT_MANIFEST = "preprocessed_data2/dataset_manifest.csv"
DEFAULT_MODEL_PATH = "latest_checkpoint.pth"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 8
DEFAULT_MIN_SIZE = 10
DEFAULT_OUTPUT_PREVIEW = "test_predictions_preview.png"


def remove_small_objects(binary_mask, min_size=DEFAULT_MIN_SIZE):
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


def evaluate_test_set(model, loader, device, min_size=30):
    """
    Evaluates trained MONAI UNet model on unseen Test split in standard FP32 precision.
    Computes 2D Tumor Slice Dice, 3D Patient Volumetric Dice, Precision, Recall, and IoU.
    """
    model.eval()

    patient_3d_preds = {}
    patient_3d_gts = {}
    pos_slice_dices = []

    total_intersection = 0.0
    total_pred_mask = 0.0
    total_gt_mask = 0.0

    print(f"Evaluating MONAI 2D UNet model on Test Set (Component filter ≥{min_size}px)...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Evaluating Test Set]"):
            images, masks, pids, s_idxs = batch[0], batch[1], batch[2], batch[3]
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            preds_raw = (torch.sigmoid(logits) > 0.5).float()

            preds_np = preds_raw.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for b in range(preds_np.shape[0]):
                p_mask = preds_np[b, 0]
                g_mask = masks_np[b, 0]
                pid = pids[b]
                s_idx = int(s_idxs[b])

                if min_size > 0:
                    p_mask = remove_small_objects(p_mask, min_size=min_size)

                # Store slice for 3D patient volume reconstruction
                if pid not in patient_3d_preds:
                    patient_3d_preds[pid] = {}
                    patient_3d_gts[pid] = {}
                patient_3d_preds[pid][s_idx] = p_mask
                patient_3d_gts[pid][s_idx] = g_mask

                # Global voxel counters
                inter = np.sum(p_mask * g_mask)
                p_sum = np.sum(p_mask)
                g_sum = np.sum(g_mask)

                total_intersection += inter
                total_pred_mask += p_sum
                total_gt_mask += g_sum

                # 2D Slice Dice on positive tumor slices
                if g_sum > 0:
                    dice_2d = (2.0 * inter) / (p_sum + g_sum + 1e-8)
                    pos_slice_dices.append(dice_2d)

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

    mean_2d_dice = float(np.mean(pos_slice_dices)) if pos_slice_dices else 0.0
    mean_3d_dice = float(np.mean(patient_3d_dices)) if patient_3d_dices else 0.0

    precision = total_intersection / (total_pred_mask + 1e-8)
    recall = total_intersection / (total_gt_mask + 1e-8)
    iou = total_intersection / (total_pred_mask + total_gt_mask - total_intersection + 1e-8)

    return mean_2d_dice, mean_3d_dice, precision, recall, iou


def save_visual_predictions(model, test_dataset, device, output_path="test_predictions_preview.png", num_samples=6, min_size=30):
    """
    Saves visual side-by-side comparison PNG previews:
    [Input CT Image] vs [Ground Truth Mask] vs [Model Prediction Overlay]
    """
    model.eval()
    pos_indices = [i for i, row in test_dataset.data_entries.iterrows() if row["has_tumor"] == 1]

    if not pos_indices:
        print("No positive tumor slices found in test set for visual preview.")
        return

    sample_indices = pos_indices[:num_samples]
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3.8 * num_samples))
    fig.suptitle("MONAI 2D UNet Tumor Predictions (Test Set)", fontsize=14, fontweight='bold')

    with torch.no_grad():
        for row_idx, data_idx in enumerate(sample_indices):
            image_tensor, mask_tensor, pid, slice_idx = test_dataset[data_idx]
            img_input = image_tensor.unsqueeze(0).to(device)

            logits = model(img_input)
            pred_mask = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()
            if min_size > 0:
                pred_mask = remove_small_objects(pred_mask, min_size=min_size)

            img_np = image_tensor.squeeze().numpy()
            gt_np = mask_tensor.squeeze().numpy()

            ax_img = axes[row_idx, 0] if num_samples > 1 else axes[0]
            ax_gt = axes[row_idx, 1] if num_samples > 1 else axes[1]
            ax_pred = axes[row_idx, 2] if num_samples > 1 else axes[2]

            # Column 1: CT Image
            ax_img.imshow(img_np, cmap='gray', vmin=0.0, vmax=1.0)
            ax_img.set_title(f"{pid} - Slice {slice_idx}\nInput CT Image", fontsize=10)
            ax_img.axis('off')

            # Column 2: Ground Truth Overlay
            ax_gt.imshow(img_np, cmap='gray', vmin=0.0, vmax=1.0)
            if np.sum(gt_np) > 0:
                ax_gt.imshow(np.ma.masked_where(gt_np == 0, gt_np), cmap='spring', alpha=0.6)
            ax_gt.set_title("Ground Truth Consensus Mask", fontsize=10, color='magenta')
            ax_gt.axis('off')

            # Column 3: Model Prediction Overlay
            ax_pred.imshow(img_np, cmap='gray', vmin=0.0, vmax=1.0)
            if np.sum(pred_mask) > 0:
                ax_pred.imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap='cool', alpha=0.6)
            ax_pred.set_title(f"UNet Prediction (≥{min_size}px)", fontsize=10, color='cyan')
            ax_pred.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visual predictions preview to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained MONAI 2D UNet Model on Test Set")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help=f"Path to manifest CSV (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to trained model checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help=f"DataLoader num_workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--min_size", type=int, default=DEFAULT_MIN_SIZE, help=f"Minimum connected component size (in pixels) to keep (default: {DEFAULT_MIN_SIZE})")
    parser.add_argument("--output_preview", type=str, default=DEFAULT_OUTPUT_PREVIEW, help=f"Path for prediction PNG (default: {DEFAULT_OUTPUT_PREVIEW})")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint file '{args.model_path}' not found. Please train a model first using train.py.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # Load Checkpoint & Instantiate MONAI 2D UNet
    checkpoint = torch.load(args.model_path, map_location=device)
    val_dice_str = f" (Best Val Dice: {checkpoint.get('val_dice', 0.0):.4f})" if 'val_dice' in checkpoint else ""
    print(f"Loading trained MONAI 2D UNet model from {args.model_path}{val_dice_str}...")

    state_dict = checkpoint["model_state_dict"]
    if "model_kwargs" in checkpoint:
        model_kwargs = checkpoint["model_kwargs"]
    else:
        first_layer_key = [k for k in state_dict.keys() if "weight" in k][0]
        base = state_dict[first_layer_key].shape[0]
        model_kwargs = {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": tuple(base * (2**i) for i in range(5)),
            "strides": (2, 2, 2, 2),
            "num_res_units": 2
        }

    model = UNet(**model_kwargs).to(device)
    model.load_state_dict(state_dict)

    _, val_transforms = get_transforms()
    test_dataset = LIDC2DDataset(args.manifest, split="test", transform=val_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    mean_2d_dice, mean_3d_dice, precision, recall, iou = evaluate_test_set(model, test_loader, device, min_size=args.min_size)

    print(f"\n=======================================================")
    print(f"TEST SET EVALUATION RESULTS (Component Filter ≥{args.min_size}px):")
    print(f"  - 2D Tumor Slice Dice:      {mean_2d_dice:.4f}")
    print(f"  - 3D Patient Volumetric Dice:{mean_3d_dice:.4f}")
    print(f"  - Intersection/Union (IoU): {iou:.4f}")
    print(f"  - Precision:                {precision:.4f}")
    print(f"  - Recall:                   {recall:.4f}")
    print(f"=======================================================")

    save_visual_predictions(model, test_dataset, device, output_path=args.output_preview, min_size=args.min_size)

if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
