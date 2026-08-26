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

# PyTorch & MONAI
import torch
import torch.nn.functional as F
import monai
from monai.networks.nets import UNet, AttentionUnet, SegResNet

# 3rd party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import inspect
from scipy.ndimage import label, find_objects

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train_2_5d import get_model
from evaluation.evaluate_2_5d import component_elongation

# Default Configuration Constants
DEFAULT_PATIENT_ID = "LIDC-IDRI-0001"
DEFAULT_MANIFEST = "preprocessed_data/dataset_manifest.csv"
DEFAULT_MODEL_PATH = "models/attention_unet_2.5d/attention_unet_2.5d.pth"
DEFAULT_MIN_VOXELS_3D = 35
DEFAULT_MIN_PEAK_PROB = 0.9985
DEFAULT_MAX_ELONGATION = 2.5
DEFAULT_TTA = 0
DEFAULT_THRESHOLD = 0.6


def remove_small_objects_3d(vol_binary, min_voxels=35, vol_prob=None, min_peak_prob=0.0,
                            max_elongation=0.0, spacing=(1.0, 1.0, 1.0)):
    """
    Applies 3D connected-component labeling on full 3D CT volume (26-connectivity).
    - Removes any 3D connected component with volume < min_voxels.
    - If min_peak_prob > 0 (and vol_prob is supplied): removes 3D components whose PEAK sigmoid
      probability never reaches min_peak_prob.
    - If max_elongation > 0: removes 3D components more elongated than that (tubular vessels),
      measured in millimetres via `spacing`.

    Mirrors evaluation/evaluate_2_5d.py so the picture matches the reported metrics.
    """
    if np.sum(vol_binary) == 0:
        return vol_binary

    use_peak_gate = (min_peak_prob > 0.0 and vol_prob is not None)
    use_shape_gate = (max_elongation > 0.0)

    # Step 1: 3D Connected-Component Size, Peak-Confidence & Shape Filtering
    if min_voxels > 0 or use_peak_gate or use_shape_gate:
        labeled_mask, num_features = label(vol_binary, structure=np.ones((3, 3, 3), dtype=bool))
        if num_features == 0:
            return vol_binary
        component_sizes = np.bincount(labeled_mask.ravel(), minlength=num_features + 1)
        too_small = np.zeros(num_features + 1, dtype=bool)

        if min_voxels > 0:
            too_small |= (component_sizes < min_voxels)

        if use_peak_gate:
            flat_labels = labeled_mask.ravel()
            fg = np.flatnonzero(flat_labels)
            comp_peak = np.zeros(num_features + 1, dtype=np.float32)
            np.maximum.at(comp_peak, flat_labels[fg], vol_prob.ravel()[fg].astype(np.float32))
            too_small |= (comp_peak < min_peak_prob)

        if use_shape_gate:
            too_small |= (component_elongation(labeled_mask, num_features, spacing) > max_elongation)

        too_small[0] = False  # Ensure background is never removed
        cleaned_vol = vol_binary.copy()
        cleaned_vol[too_small[labeled_mask]] = False
    else:
        cleaned_vol = vol_binary.copy()

    return cleaned_vol


def predict_probs(model, images, tta=False):
    """
    Sigmoid probabilities for a batch. With tta=True the prediction is averaged over the
    4 horizontal/vertical flip combinations, each flip undone before accumulating.
    Training used RandFlipd on both spatial axes, so these views are in-distribution.

    Mirrors evaluation/evaluate_2_5d.py.
    """
    views = [()] if not tta else [(), (2,), (3,), (2, 3)]
    acc = None
    for dims in views:
        x = torch.flip(images, dims) if dims else images
        logits = model(x)
        p = torch.sigmoid(logits.float())
        if dims:
            p = torch.flip(p, dims)
        acc = p if acc is None else acc + p
    return acc / len(views)


def compute_slice_dice(pred_mask, gt_mask):
    """
    Computes 2D Dice score for a single slice.
    Returns 1.0 if both masks are empty.
    """
    intersection = np.sum(pred_mask * gt_mask)
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)

    if pred_sum == 0 and gt_sum == 0:
        return 1.0  # Perfect match on negative slice

    return (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)


def load_patient_slices(manifest_path, patient_id):
    """
    Loads all preprocessed 2.5D slice metadata for a specified patient ID from manifest.
    """
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest CSV file '{manifest_path}' not found.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    df["filepath"] = df["filepath"].str.replace("\\", "/", regex=False)

    # Support shorthand integer patient ID (e.g., -p 72 -> LIDC-IDRI-0072)
    if patient_id.isdigit():
        patient_id = f"LIDC-IDRI-{int(patient_id):04d}"

    patient_df = df[df["patient_id"] == patient_id].sort_values(by="slice_idx").reset_index(drop=True)

    if len(patient_df) == 0:
        available_patients = df["patient_id"].unique()
        print(f"\nError: Patient ID '{patient_id}' not found in manifest!")
        print(f"Available patients in dataset ({len(available_patients)} total):")
        print(f"  {', '.join(available_patients[:15])} ...")
        sys.exit(1)

    return patient_df, patient_id


def load_trained_model(model_path, device):
    """
    Loads trained 2.5D MONAI model checkpoint (auto-detects UNet vs AttentionUnet vs SegResNet).
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Train first using train_2_5d.py.")
        sys.exit(1)

    print(f"Loading trained 2.5D MONAI model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if "model_type" in checkpoint:
        try:
            model, _ = get_model(checkpoint["model_type"], in_channels=3)
            model.load_state_dict(state_dict)
            model.eval()
            return model.to(device)
        except Exception:
            pass

    if "model_kwargs" in checkpoint and isinstance(checkpoint["model_kwargs"], dict):
        kwargs = checkpoint["model_kwargs"].copy()
        for model_cls in [UNet, AttentionUnet, SegResNet]:
            try:
                valid_params = inspect.signature(model_cls.__init__).parameters.keys()
                filtered = {k: v for k, v in kwargs.items() if k in valid_params}
                model = model_cls(**filtered)
                model.load_state_dict(state_dict)
                model.eval()
                return model.to(device)
            except Exception:
                continue

    for m_type in ["unet", "attention_unet", "segresnet"]:
        try:
            model, _ = get_model(m_type, in_channels=3)
            model.load_state_dict(state_dict)
            model.eval()
            return model.to(device)
        except Exception:
            continue

    raise RuntimeError(f"Failed to load checkpoint from {model_path} into any supported 2.5D architecture.")


class PatientSliceVisualizer25D:
    def __init__(self, patient_df, model, device, patient_id, min_voxels_3d=35,
                 threshold=0.5, min_peak_prob=0.0, max_elongation=0.0, tta=False):
        self.patient_df = patient_df
        self.model = model
        self.device = device
        self.patient_id = patient_id
        self.min_voxels_3d = min_voxels_3d
        self.threshold = threshold
        self.min_peak_prob = min_peak_prob
        self.max_elongation = max_elongation
        self.tta = tta
        self.total_slices = len(patient_df)
        self.current_idx = 0

        filter_parts = [f"≥{min_voxels_3d} voxels"]
        if min_peak_prob > 0.0:
            filter_parts.append(f"peak ≥{min_peak_prob:g}")
        if max_elongation > 0.0:
            filter_parts.append(f"elong ≤{max_elongation:g}")
        if tta:
            filter_parts.append("4-flip TTA")
        self.filter_str = ", ".join(filter_parts)

        print(f"\nPre-computing predictions for all {self.total_slices} slices of patient {patient_id} (3D Filter: {self.filter_str}, Threshold: {threshold})...")
        self.images = []
        self.gt_masks = []
        raw_probs = []
        self.slice_indices = []
        self.gt_tumor_flags = []

        slice_map = dict(zip(patient_df["slice_idx"], patient_df["filepath"]))
        max_z = patient_df["slice_idx"].max()

        with torch.no_grad():
            for idx, row in patient_df.iterrows():
                z = int(row["slice_idx"])
                z_prev = max(0, z - 1)
                z_curr = z
                z_next = min(max_z, z + 1)

                path_prev = slice_map.get(z_prev, slice_map[z_curr])
                path_curr = slice_map[z_curr]
                path_next = slice_map.get(z_next, slice_map[z_curr])

                if not (os.path.exists(path_prev) and os.path.exists(path_curr) and os.path.exists(path_next)):
                    continue

                npz_prev = np.load(path_prev)
                npz_curr = np.load(path_curr)
                npz_next = np.load(path_next)

                img_prev = npz_prev["image"].astype(np.float32)
                img_curr = npz_curr["image"].astype(np.float32)
                gt = npz_curr["mask"].astype(np.float32)
                img_next = npz_next["image"].astype(np.float32)

                img_3ch = np.concatenate([img_prev, img_curr, img_next], axis=0)  # (3, H, W)
                orig_shape = img_curr.shape[-2:]

                # Resize input tensor to (256, 256) for UNet execution
                img_tensor = torch.from_numpy(img_3ch).unsqueeze(0).to(device)  # (1, 3, H, W)
                img_resized = F.interpolate(img_tensor, size=(256, 256), mode='bilinear', align_corners=False)

                prob_resized = predict_probs(model, img_resized, tta=self.tta)

                # Interpolate the probability back to original (H, W) resolution
                prob_t = F.interpolate(prob_resized, size=orig_shape, mode='bilinear', align_corners=False)
                prob = prob_t.squeeze().cpu().numpy().astype(np.float32)

                self.images.append(img_curr.squeeze())
                self.gt_masks.append(gt.squeeze())
                raw_probs.append(prob)
                self.slice_indices.append(z)
                self.gt_tumor_flags.append(int(row["has_tumor"]))

        self.total_slices = len(self.images)
        if self.total_slices == 0:
            print("Error: No valid slice files were loaded.")
            sys.exit(1)

        # Assemble full 3D probability volume, binarize and apply 3D Volumetric Filtering + Peak-Confidence Gate
        vol_prob = np.stack(raw_probs, axis=-1)
        vol_pred = vol_prob > self.threshold
        # The evaluator gates on the 256x256 grid; here the volume is at the patient's
        # native 1mm crop, so a component holds more voxels. Rescale by the area ratio
        # so the same --min_voxels_3d means the same physical size in both tools.
        area_ratio = (vol_prob.shape[0] * vol_prob.shape[1]) / (256.0 * 256.0)
        min_vox_here = int(round(self.min_voxels_3d * area_ratio))
        if min_vox_here != self.min_voxels_3d:
            print(f"  min_voxels_3d {self.min_voxels_3d} (256 grid) -> {min_vox_here} "
                  f"on this patient's {vol_prob.shape[0]}x{vol_prob.shape[1]} native grid")
        if (min_vox_here > 0 or self.min_peak_prob > 0.0 or self.max_elongation > 0.0) and np.sum(vol_pred) > 0:
            vol_pred = remove_small_objects_3d(
                vol_pred,
                min_voxels=min_vox_here,
                vol_prob=vol_prob,
                min_peak_prob=self.min_peak_prob,
                max_elongation=self.max_elongation,
                spacing=(1.0, 1.0, 1.0),   # npz grid is 1mm isotropic after preprocessing
            )

        self.pred_masks = [vol_pred[:, :, i].astype(np.float32) for i in range(self.total_slices)]

        print(f"Successfully loaded {self.total_slices} slices for patient {patient_id}.")
        print("Starting interactive GUI... Use ← / → Arrow Keys or ON-SCREEN BUTTONS to navigate.")

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
        self.axes[0].set_title(f"Input CT Image\nSlice {slice_num} ({self.current_idx + 1}/{self.total_slices}) | Shape: {img_np.shape}", fontsize=11, fontweight='bold')
        self.axes[0].axis('off')

        # 2. Ground Truth Mask Overlay
        self.axes[1].imshow(img_np, cmap='gray', vmin=0.0, vmax=1.0)
        if gt_pixels > 0:
            self.axes[1].imshow(np.ma.masked_where(gt_np == 0, gt_np), cmap='spring', alpha=0.6)
            gt_status = f"[TUMOR PRESENT] ({gt_pixels} px)"
            gt_color = 'crimson'
        else:
            gt_status = "[HEALTHY LUNG] (0 px)"
            gt_color = 'darkgreen'
        self.axes[1].set_title(f"Ground Truth Consensus Mask\n{gt_status}", fontsize=11, fontweight='bold', color=gt_color)
        self.axes[1].axis('off')

        # 3. Model 2.5D UNet Prediction Overlay
        self.axes[2].imshow(img_np, cmap='gray', vmin=0.0, vmax=1.0)
        if pred_pixels > 0:
            self.axes[2].imshow(np.ma.masked_where(pred_np == 0, pred_np), cmap='cool', alpha=0.6)
            pred_status = f"[PREDICTED TUMOR] ({pred_pixels} px)"
            pred_color = 'darkcyan'
        else:
            pred_status = "[PREDICTED HEALTHY]"
            pred_color = 'darkgreen'
        filter_str = f" [{self.filter_str}]" if self.filter_str else ""
        self.axes[2].set_title(f"MONAI 2.5D UNet Prediction{filter_str}\n{pred_status} | Dice: {slice_dice:.3f}", fontsize=11, fontweight='bold', color=pred_color)
        self.axes[2].axis('off')

        # Overall Figure Title
        overall_status = "TUMOR SLICE" if has_gt_tumor else "HEALTHY SLICE"
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
    parser = argparse.ArgumentParser(description="Interactive Patient 2.5D Slice Visualizer (CT vs Ground Truth vs MONAI 2.5D UNet)")
    parser.add_argument("-p", "--patient_id", type=str, default=DEFAULT_PATIENT_ID, help=f"Patient ID to visualize (e.g. LIDC-IDRI-0072 or 72) (default: {DEFAULT_PATIENT_ID})")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help=f"Path to preprocessed dataset manifest CSV (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to trained model checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--min_voxels_3d", "--min_voxels", "--min_size", type=int, default=DEFAULT_MIN_VOXELS_3D, help=f"Minimum 3D connected component volume in voxels (default: {DEFAULT_MIN_VOXELS_3D})")
    parser.add_argument("--min_peak_prob", "--peak_gate", type=float, default=DEFAULT_MIN_PEAK_PROB, help=f"Keep a 3D connected component only if its peak sigmoid probability reaches this value (default: {DEFAULT_MIN_PEAK_PROB})")
    parser.add_argument("--max_elongation", "--max_elong", type=float, default=DEFAULT_MAX_ELONGATION,
                        help=f"Drop 3D components more elongated than this (sqrt(lambda1/lambda3) in mm); 0 disables (default: {DEFAULT_MAX_ELONGATION})")
    parser.add_argument("--tta", type=int, default=DEFAULT_TTA,
                        help=f"1 = average predictions over the 4 horizontal/vertical flips (default: {DEFAULT_TTA})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Probability threshold for binarizing predictions (default: {DEFAULT_THRESHOLD})")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    patient_df, patient_id = load_patient_slices(args.manifest, args.patient_id)
    model = load_trained_model(args.model_path, device)

    visualizer = PatientSliceVisualizer25D(
        patient_df, model, device, patient_id,
        min_voxels_3d=args.min_voxels_3d,
        threshold=args.threshold,
        min_peak_prob=args.min_peak_prob,
        max_elongation=args.max_elongation,
        tta=bool(args.tta),
    )
    plt.show()


if __name__ == "__main__":
    main()
