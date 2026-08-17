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

# PyTorch & MONAI
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.networks.nets import UNet, AttentionUnet, SegResNet

# 3rd party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_erosion, find_objects
from tqdm import tqdm

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train import LIDC2DDataset, get_transforms

# Default Configuration Constants
DEFAULT_MANIFEST = "preprocessed_data/dataset_manifest.csv"
DEFAULT_MODEL_PATH = "models/unet/unet.pth"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 8
DEFAULT_MIN_SIZE = 10
DEFAULT_REPORT_PATH = "models/unet/test_evaluation_report.txt"


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

    # Distance matrix using broadcasting
    d_p2g = np.min(np.linalg.norm(pred_pts[:, None, :] - gt_pts[None, :, :], axis=2), axis=1)
    d_g2p = np.min(np.linalg.norm(gt_pts[:, None, :] - pred_pts[None, :, :], axis=2), axis=1)

    all_dists = np.concatenate([d_p2g, d_g2p])
    hd95 = float(np.percentile(all_dists, 95))
    asd = float(np.mean(all_dists))
    return hd95, asd


def compute_single_mask_metrics(p_mask, g_mask):
    """
    Computes Dice, IoU, Precision, Sensitivity, Specificity, HD95, ASD, and Failure flag
    with explicit handling for empty masks.
    """
    p_sum = np.sum(p_mask)
    g_sum = np.sum(g_mask)

    tp = np.sum(p_mask * g_mask)
    fp = np.sum(p_mask * (1.0 - g_mask))
    fn = np.sum((1.0 - p_mask) * g_mask)
    tn = np.sum((1.0 - p_mask) * (1.0 - g_mask))

    # Case 1: Both GT and Pred are empty (Perfect negative match)
    if g_sum == 0 and p_sum == 0:
        return {
            "dice": 1.0, "iou": 1.0, "precision": 1.0, "sensitivity": 1.0, "specificity": 1.0,
            "hd95": 0.0, "asd": 0.0, "is_failure": False, "is_empty_gt": True
        }

    # Case 2: GT is empty, but Pred contains false positives
    if g_sum == 0 and p_sum > 0:
        spec = tn / (tn + fp + 1e-8)
        return {
            "dice": 0.0, "iou": 0.0, "precision": 0.0, "sensitivity": 1.0, "specificity": spec,
            "hd95": np.nan, "asd": np.nan, "is_failure": True, "is_empty_gt": True
        }

    # Case 3: GT is non-empty, but Pred is empty (False Negative / Miss)
    if g_sum > 0 and p_sum == 0:
        return {
            "dice": 0.0, "iou": 0.0, "precision": 0.0, "sensitivity": 0.0, "specificity": 1.0,
            "hd95": np.nan, "asd": np.nan, "is_failure": True, "is_empty_gt": False
        }

    # Case 4: Both GT and Pred are non-empty
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    hd95, asd = compute_surface_distances(p_mask > 0, g_mask > 0)
    is_failure = bool(dice < 0.1)

    return {
        "dice": dice, "iou": iou, "precision": prec, "sensitivity": sens, "specificity": spec,
        "hd95": hd95, "asd": asd, "is_failure": is_failure, "is_empty_gt": False
    }


def aggregate_metric_dict(list_of_metric_dicts):
    """
    Summarizes a list of metric dicts into mean values and failure rates.
    """
    if not list_of_metric_dicts:
        return {
            "dice": 0.0, "iou": 0.0, "precision": 0.0, "sensitivity": 0.0, "specificity": 0.0,
            "hd95": 0.0, "asd": 0.0, "failure_rate": 0.0, "count": 0
        }

    dices = [m["dice"] for m in list_of_metric_dicts]
    ious = [m["iou"] for m in list_of_metric_dicts]
    precs = [m["precision"] for m in list_of_metric_dicts]
    senss = [m["sensitivity"] for m in list_of_metric_dicts]
    specs = [m["specificity"] for m in list_of_metric_dicts]

    hd95s = [m["hd95"] for m in list_of_metric_dicts if not np.isnan(m["hd95"])]
    asds = [m["asd"] for m in list_of_metric_dicts if not np.isnan(m["asd"])]
    failures = [1.0 if m["is_failure"] else 0.0 for m in list_of_metric_dicts]

    return {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "sensitivity": float(np.mean(senss)),
        "specificity": float(np.mean(specs)),
        "hd95": float(np.mean(hd95s)) if hd95s else 0.0,
        "asd": float(np.mean(asds)) if asds else 0.0,
        "failure_rate": float(np.mean(failures)),
        "count": len(list_of_metric_dicts)
    }


def evaluate_test_set_hierarchical(model, loader, device, min_size):
    """
    Comprehensive hierarchical evaluation:
    1. Per-Slice evaluation (All slices & Positive tumor slices)
    2. Per-Patient evaluation (Full 3D patient volume reconstruction)
    3. Per-Nodule 3D evaluation & size stratification (Small, Medium, Large nodules)
    """
    model.eval()

    patient_3d_preds = {}
    patient_3d_gts = {}

    slice_metrics_all = []
    slice_metrics_pos = []

    print(f"\nRunning Comprehensive Evaluation on Test Set (Component filter ≥{min_size}px)...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Evaluating Test Slices]"):
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

                # Store for 3D patient reconstruction
                if pid not in patient_3d_preds:
                    patient_3d_preds[pid] = {}
                    patient_3d_gts[pid] = {}
                patient_3d_preds[pid][s_idx] = p_mask
                patient_3d_gts[pid][s_idx] = g_mask

                # 1. Per-Slice Metric Computation
                m_slice = compute_single_mask_metrics(p_mask, g_mask)
                slice_metrics_all.append(m_slice)
                if not m_slice["is_empty_gt"]:
                    slice_metrics_pos.append(m_slice)

    # 2. Per-Patient 3D Volume Evaluation & Per-Patient Row Collection
    patient_metrics_all = []
    patient_metrics_pos = []
    per_patient_rows = []

    # 3. Per-Nodule 3D Evaluation & Size Stratification
    nodule_metrics_all = []
    nodule_metrics_small = []   # < 100 voxels
    nodule_metrics_medium = []  # 100 - 1000 voxels
    nodule_metrics_large = []   # >= 1000 voxels

    for pid in tqdm(sorted(patient_3d_preds.keys()), desc="[3D Patient & Nodule Reconstruction]"):
        s_keys = sorted(patient_3d_preds[pid].keys())
        vol_pred = np.stack([patient_3d_preds[pid][k] for k in s_keys], axis=-1)
        vol_gt = np.stack([patient_3d_gts[pid][k] for k in s_keys], axis=-1)

        m_patient = compute_single_mask_metrics(vol_pred, vol_gt)
        patient_metrics_all.append(m_patient)
        if not m_patient["is_empty_gt"]:
            patient_metrics_pos.append(m_patient)

        # Compute 2D slice metrics for this patient (both tumor slices and all slices)
        patient_2d_tumor_metrics = []
        patient_2d_all_metrics = []
        has_tumor_scan = (np.sum(vol_gt) > 0)
        num_tumor_slices = 0

        for k in s_keys:
            p_slice = patient_3d_preds[pid][k]
            g_slice = patient_3d_gts[pid][k]
            m_2d = compute_single_mask_metrics(p_slice, g_slice)
            patient_2d_all_metrics.append(m_2d)

            if np.sum(g_slice) > 0:
                num_tumor_slices += 1
                patient_2d_tumor_metrics.append(m_2d)

        agg_tumor_2d = aggregate_metric_dict(patient_2d_tumor_metrics)
        agg_all_2d = aggregate_metric_dict(patient_2d_all_metrics)

        per_patient_rows.append({
            "patient_id": pid,
            "total_slices": len(s_keys),
            "tumor_slices": num_tumor_slices,
            "has_tumor": 1 if has_tumor_scan else 0,
            "dice_2d_tumor": agg_tumor_2d["dice"] if has_tumor_scan else 1.0,
            "dice_2d_all": agg_all_2d["dice"],
            "iou_2d_tumor": agg_tumor_2d["iou"] if has_tumor_scan else 1.0,
            "precision_2d_tumor": agg_tumor_2d["precision"] if has_tumor_scan else 1.0,
            "sensitivity_2d_tumor": agg_tumor_2d["sensitivity"] if has_tumor_scan else 1.0,
            "specificity_2d_all": agg_all_2d["specificity"],
            "hd95_2d_tumor": agg_tumor_2d["hd95"] if has_tumor_scan else 0.0,
            "asd_2d_tumor": agg_tumor_2d["asd"] if has_tumor_scan else 0.0,
            "dice_3d_vol": m_patient["dice"]
        })

        # Extract individual 3D connected nodule lesions using 3D labeling
        if np.sum(vol_gt) > 0:
            labeled_gt, num_nodules = label(vol_gt > 0)
            slices = find_objects(labeled_gt)

            for n_idx, bbox in enumerate(slices, 1):
                if bbox is None:
                    continue
                nodule_gt_mask = (labeled_gt[bbox] == n_idx)
                nodule_pred_mask = vol_pred[bbox]

                vol_voxels = int(np.sum(nodule_gt_mask))
                m_nodule = compute_single_mask_metrics(nodule_pred_mask, nodule_gt_mask)
                nodule_metrics_all.append(m_nodule)

                if vol_voxels < 100:
                    nodule_metrics_small.append(m_nodule)
                elif vol_voxels < 1000:
                    nodule_metrics_medium.append(m_nodule)
                else:
                    nodule_metrics_large.append(m_nodule)

    df_patients = pd.DataFrame(per_patient_rows)

    # Aggregate summaries
    results = {
        "slice_all": aggregate_metric_dict(slice_metrics_all),
        "slice_pos": aggregate_metric_dict(slice_metrics_pos),
        "patient_all": aggregate_metric_dict(patient_metrics_all),
        "patient_pos": aggregate_metric_dict(patient_metrics_pos),
        "nodule_all": aggregate_metric_dict(nodule_metrics_all),
        "nodule_small": aggregate_metric_dict(nodule_metrics_small),
        "nodule_medium": aggregate_metric_dict(nodule_metrics_medium),
        "nodule_large": aggregate_metric_dict(nodule_metrics_large),
        "per_patient_df": df_patients
    }

    return results


def print_and_format_report(results, min_size, report_path):
    """
    Formats clean metric tables and saves full report to file.
    """
    lines = []
    lines.append("=========================================================================================================")
    lines.append(f"                   HIERARCHICAL EVALUATION REPORT (Filter ≥{min_size}px)")
    lines.append("=========================================================================================================\n")

    def format_row(title, m):
        return (
            f"  {title:<32s} | Dice: {m['dice']:.4f} | IoU: {m['iou']:.4f} | "
            f"Prec: {m['precision']:.4f} | Sens: {m['sensitivity']:.4f} | Spec: {m['specificity']:.4f} | "
            f"HD95: {m['hd95']:5.2f}mm | ASD: {m['asd']:5.2f}mm | Fail: {m['failure_rate']*100:5.1f}% | Count: {m['count']:4d}"
        )

    lines.append("1. PER-SLICE EVALUATION:")
    lines.append("---------------------------------------------------------------------------------------------------------")
    lines.append(format_row("2D Slices (Tumor Slices Only)", results["slice_pos"]))
    lines.append(format_row("2D Slices (All Test Slices)", results["slice_all"]))
    lines.append("")

    lines.append("2. PER-PATIENT 3D RECONSTRUCTION EVALUATION:")
    lines.append("---------------------------------------------------------------------------------------------------------")
    lines.append(format_row("3D Patients (Nodule Scans Only)", results["patient_pos"]))
    lines.append(format_row("3D Patients (All Test Patients)", results["patient_all"]))
    lines.append("")

    lines.append("3. PER-NODULE 3D LESION & SIZE STRATIFICATION:")
    lines.append("---------------------------------------------------------------------------------------------------------")
    lines.append(format_row("3D Nodules (All Nodule Lesions)", results["nodule_all"]))
    lines.append(format_row("Small Nodules  (< 100 voxels)", results["nodule_small"]))
    lines.append(format_row("Medium Nodules (100 - 1000 voxels)", results["nodule_medium"]))
    lines.append(format_row("Large Nodules  (>= 1000 voxels)", results["nodule_large"]))
    lines.append("")

    if "per_patient_df" in results:
        df_p = results["per_patient_df"]
        lines.append("4. INDIVIDUAL PER-PATIENT BREAKDOWN SAMPLE (Tumor Scans - 2D & 3D Metrics):")
        lines.append("---------------------------------------------------------------------------------------------------------")
        lines.append("  Patient ID          | Slices (Tumor) | 2D Tumor Dice | 2D All Dice | 2D Tum IoU | 2D Tum Prec | 2D Tum Sens | 3D Vol Dice")
        lines.append("  -----------------------------------------------------------------------------------------------------------------------")
        tumor_patients = df_p[df_p["has_tumor"] == 1].head(15)
        for _, r in tumor_patients.iterrows():
            lines.append(
                f"  {r['patient_id']:<19s} | {r['total_slices']:3d} ({r['tumor_slices']:3d})   | "
                f"{r['dice_2d_tumor']:.4f}        | {r['dice_2d_all']:.4f}      | {r['iou_2d_tumor']:.4f}     | "
                f"{r['precision_2d_tumor']:.4f}      | {r['sensitivity_2d_tumor']:.4f}      | {r['dice_3d_vol']:.4f}"
            )
        lines.append("  (Full table exported to patient_evaluation_breakdown.csv)")
    lines.append("=========================================================================================================\n")

    report_text = "\n".join(lines)
    print(report_text)

    report_dir = os.path.dirname(os.path.abspath(report_path))
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved evaluation report to: {report_path}")

    if "per_patient_df" in results:
        report_filename = os.path.basename(report_path)
        report_stem, _ = os.path.splitext(report_filename)
        if report_stem == "test_evaluation_report":
            csv_name = "patient_evaluation_breakdown.csv"
        else:
            csv_name = f"patient_breakdown_{report_stem}.csv"
        breakdown_path = os.path.join(report_dir, csv_name)
        results["per_patient_df"].to_csv(breakdown_path, index=False)
        print(f"Saved individual per-patient breakdown to: {breakdown_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hierarchical Evaluation of MONAI 2D UNet")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help=f"Path to manifest CSV (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to trained model checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help=f"DataLoader num_workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--min_size", type=int, default=DEFAULT_MIN_SIZE, help=f"Minimum connected component size (in pixels) to keep (default: {DEFAULT_MIN_SIZE})")
    parser.add_argument("--report_path", type=str, default=DEFAULT_REPORT_PATH, help=f"Path for evaluation text report (default: {DEFAULT_REPORT_PATH})")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint file '{args.model_path}' not found. Please specify valid model path.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location=device)
    val_dice_str = f" (Val Dice: {checkpoint.get('val_dice', 0.0):.4f})" if 'val_dice' in checkpoint else ""
    print(f"Loading MONAI model from {args.model_path}{val_dice_str}...")

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

    # Dynamically detect UNet vs AttentionUnet vs SegResNet architecture from checkpoint weights
    try:
        model = UNet(**model_kwargs).to(device)
        model.load_state_dict(state_dict)
    except Exception:
        try:
            model = AttentionUnet(**model_kwargs).to(device)
            model.load_state_dict(state_dict)
        except Exception:
            model = SegResNet(**model_kwargs).to(device)
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

    results = evaluate_test_set_hierarchical(model, test_loader, device, min_size=args.min_size)
    print_and_format_report(results, args.min_size, report_path=args.report_path)

if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
