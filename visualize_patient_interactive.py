# Standard library
import os
import sys
import argparse

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
from matplotlib.widgets import Button
from scipy.ndimage import label

# PyTorch & MONAI
import torch
import monai
from monai.networks.nets import AttentionUnet, UNet

# Default Configuration Constants
DEFAULT_PATIENT_ID = "LIDC-IDRI-0072"
DEFAULT_MANIFEST = "preprocessed_data/dataset_manifest.csv"
DEFAULT_MODEL_PATH = "models/attention_unet/attention_unet.pth"
DEFAULT_MIN_SIZE = 30

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

def compute_slice_dice(pred_mask, gt_mask):
    """
    Computes 2D Dice score for a single slice.
    Returns 1.0 if both masks are empty, 0.0 if one is empty and other is non-empty.
    """
    intersection = np.sum(pred_mask * gt_mask)
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)
    
    if pred_sum == 0 and gt_sum == 0:
        return 1.0  # Perfect match on negative slice
    
    return (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)

def load_patient_slices(manifest_path, patient_id):
    """
    Loads all preprocessed 2D slice metadata for a specified patient ID from manifest.
    """
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest CSV file '{manifest_path}' not found.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    df["filepath"] = df["filepath"].str.replace("\\", "/", regex=False)

    patient_df = df[df["patient_id"] == patient_id].sort_values(by="slice_idx").reset_index(drop=True)

    if len(patient_df) == 0:
        available_patients = df["patient_id"].unique()
        print(f"\nError: Patient ID '{patient_id}' not found in manifest!")
        print(f"Available patients in dataset ({len(available_patients)} total):")
        print(f"  {', '.join(available_patients[:15])} ...")
        sys.exit(1)

    return patient_df

def load_trained_model(model_path, device):
    """
    Loads trained 2D MONAI UNet model checkpoint, auto-detecting model architecture channels.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Train first using train_monai_2d.py.")
        sys.exit(1)

    print(f"Loading trained MONAI UNet model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    if "model_kwargs" in checkpoint:
        model_kwargs = checkpoint["model_kwargs"]
    else:
        # Auto-detect base channel size from first conv layer weight shape
        first_layer_key = [k for k in state_dict.keys() if "weight" in k][0]
        base = state_dict[first_layer_key].shape[0]
        model_kwargs = {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": tuple(base * (2**i) for i in range(5)),
            "strides": (2, 2, 2, 2)
        }

    try:
        model = AttentionUnet(**model_kwargs).to(device)
        model.load_state_dict(state_dict)
    except Exception:
        model = UNet(**model_kwargs).to(device)
        model.load_state_dict(state_dict)
    model.eval()
    return model


class PatientSliceVisualizer:
    def __init__(self, patient_df, model, device, patient_id, min_size=30):
        self.patient_df = patient_df
        self.model = model
        self.device = device
        self.patient_id = patient_id
        self.min_size = min_size
        self.total_slices = len(patient_df)
        self.current_idx = 0

        print(f"\nPre-computing predictions for all {self.total_slices} slices of patient {patient_id} (Filtering objects < {min_size}px)...")
        self.images = []
        self.gt_masks = []
        self.pred_masks = []
        self.slice_indices = []
        self.gt_tumor_flags = []

        with torch.no_grad():
            for idx, row in patient_df.iterrows():
                filepath = row["filepath"]
                if not os.path.exists(filepath):
                    print(f"Warning: File {filepath} not found, skipping...")
                    continue

                npz_data = np.load(filepath)
                img = npz_data["image"].astype(np.float32)  # (1, 512, 512)
                gt = npz_data["mask"].astype(np.float32)    # (1, 512, 512)

                img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, 1, 512, 512)
                logits = model(img_tensor)
                pred = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()
                
                # Remove small component noise (< min_size px)
                if self.min_size > 0:
                    pred = remove_small_objects(pred, min_size=self.min_size)

                self.images.append(img.squeeze())
                self.gt_masks.append(gt.squeeze())
                self.pred_masks.append(pred)
                self.slice_indices.append(int(row["slice_idx"]))
                self.gt_tumor_flags.append(int(row["has_tumor"]))

        self.total_slices = len(self.images)
        if self.total_slices == 0:
            print("Error: No valid slice files were loaded.")
            sys.exit(1)

        print(f"Successfully loaded {self.total_slices} slices for patient {patient_id}.")
        print("Starting interactive GUI... Use LEFT / RIGHT arrow keys or ON-SCREEN BUTTONS to navigate.")

        # Setup Figure & Axes
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 6))
        self.fig.canvas.manager.set_window_title(f"LIDC-IDRI Patient Visualizer - {patient_id}")
        plt.subplots_adjust(bottom=0.15, top=0.88, left=0.05, right=0.95, wspace=0.15)

        # Add Previous / Next GUI Buttons
        ax_prev = plt.axes([0.35, 0.03, 0.12, 0.06])
        ax_next = plt.axes([0.53, 0.03, 0.12, 0.06])
        
        self.btn_prev = Button(ax_prev, '◀ Previous', color='lightgray', hovercolor='0.85')
        self.btn_next = Button(ax_next, 'Next ▶', color='lightgray', hovercolor='0.85')

        self.btn_prev.on_clicked(self.prev_slice)
        self.btn_next.on_clicked(self.next_slice)

        # Connect Keyboard Events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Initial Plot Render
        self.update_plot()

    def update_plot(self):
        img_np = self.images[self.current_idx]
        gt_np = self.gt_masks[self.current_idx]
        pred_np = self.pred_masks[self.current_idx]
        slice_num = self.slice_indices[self.current_idx]
        has_gt_tumor = self.gt_tumor_flags[self.current_idx] == 1

        gt_pixels = int(np.sum(gt_np))
        pred_pixels = int(np.sum(pred_np))
        slice_dice = compute_slice_dice(pred_np, gt_np)

        for ax in self.axes:
            ax.clear()

        # 1. Input CT Image
        self.axes[0].imshow(img_np, cmap='gray', vmin=0.0, vmax=1.0)
        self.axes[0].set_title(f"Input CT Image\nSlice {slice_num} ({self.current_idx + 1}/{self.total_slices})", fontsize=11, fontweight='bold')
        self.axes[0].axis('off')

        # 2. Ground Truth Mask Overlay
        self.axes[1].imshow(img_np, cmap='gray', vmin=0.0, vmax=1.0)
        if gt_pixels > 0:
            self.axes[1].imshow(np.ma.masked_where(gt_np == 0, gt_np), cmap='spring', alpha=0.6)
            gt_status = f"TUMOR PRESENT ({gt_pixels} px)"
            gt_color = 'magenta'
        else:
            gt_status = "No Tumor (Healthy Lung)"
            gt_color = 'green'
        self.axes[1].set_title(f"Ground Truth Consensus Mask\n{gt_status}", fontsize=11, fontweight='bold', color=gt_color)
        self.axes[1].axis('off')

        # 3. Model UNet Prediction Overlay
        self.axes[2].imshow(img_np, cmap='gray', vmin=0.0, vmax=1.0)
        if pred_pixels > 0:
            self.axes[2].imshow(np.ma.masked_where(pred_np == 0, pred_np), cmap='cool', alpha=0.6)
            pred_status = f"PREDICTED TUMOR ({pred_pixels} px)"
            pred_color = 'darkcyan'
        else:
            pred_status = "Predicted Healthy"
            pred_color = 'darkgreen'
        filter_str = f" [≥{self.min_size}px Filter]" if self.min_size > 0 else ""
        self.axes[2].set_title(f"MONAI UNet Prediction{filter_str}\n{pred_status} | Dice: {slice_dice:.3f}", fontsize=11, fontweight='bold', color=pred_color)
        self.axes[2].axis('off')

        # Overall Figure Title
        overall_status = "🔴 POSITIVE TUMOR SLICE" if has_gt_tumor else "🟢 HEALTHY SLICE"
        self.fig.suptitle(
            f"Patient: {self.patient_id}  |  Slice {self.current_idx + 1} of {self.total_slices} (CT Slice #{slice_num})  |  {overall_status}\n"
            f"Use ← / → Arrow Keys or GUI Buttons to Navigate Slices",
            fontsize=13, fontweight='bold'
        )

        self.fig.canvas.draw_idle()

    def prev_slice(self, event=None):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_plot()

    def next_slice(self, event=None):
        if self.current_idx < self.total_slices - 1:
            self.current_idx += 1
            self.update_plot()

    def on_key_press(self, event):
        if event.key in ['left', 'down', 'page-down', 'b']:
            self.prev_slice()
        elif event.key in ['right', 'up', 'page-up', 'n']:
            self.next_slice()

def main():
    parser = argparse.ArgumentParser(description="Interactive Patient 2D Slice Visualizer (CT vs Ground Truth vs MONAI UNet)")
    parser.add_argument("--patient_id", type=str, default=DEFAULT_PATIENT_ID, help=f"Patient ID to visualize (default: {DEFAULT_PATIENT_ID})")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help=f"Path to preprocessed dataset manifest CSV (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to trained model checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--min_size", type=int, default=DEFAULT_MIN_SIZE, help=f"Minimum connected component size (in pixels) to keep (default: {DEFAULT_MIN_SIZE})")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    patient_df = load_patient_slices(args.manifest, args.patient_id)
    model = load_trained_model(args.model_path, device)

    visualizer = PatientSliceVisualizer(patient_df, model, device, args.patient_id, min_size=args.min_size)
    plt.show()

if __name__ == "__main__":
    main()



