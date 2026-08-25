# Standard library
import os
import sys
import argparse
import random
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
import monai
from monai.networks.nets import UNet, AttentionUnet, SegResNet

# 3rd party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_erosion, find_objects, distance_transform_edt
from tqdm import tqdm

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train import LIDC2DDataset, get_transforms, get_model

# Default Configuration Constants
DEFAULT_MANIFEST = "preprocessed_data/dataset_manifest.csv"
DEFAULT_MODEL_PATH = "models/unet/unet.pth"
DEFAULT_SPLIT = "test"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 8
DEFAULT_MIN_VOXELS_3D = 15
DEFAULT_MIN_PEAK_PROB = 0.0
DEFAULT_MAX_ELONGATION = 0.0   # 0 = elongation filter disabled
DEFAULT_TTA = 0                # 1 = average sigmoid over 4 flips
DEFAULT_REPORT_PATH = "models/unet/test_evaluation_report.txt"
DEFAULT_THRESHOLD = 0.5
DEFAULT_SEED = 42


def set_seed(seed=42):
    """
    Sets global random seed for Python random, NumPy, PyTorch (CPU & CUDA), and MONAI
    to guarantee full evaluation reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    monai.utils.set_determinism(seed=seed)


def component_elongation(labeled_mask, num_features, spacing):
    """
    Per-component sqrt(lambda1 / lambda3) of the voxel-coordinate covariance, in millimetres.

    Pulmonary vessels are tubular and score high; nodules are compact and score near 1.
    Coordinates are scaled by `spacing` because voxels are anisotropic after the 256x256
    resize (in-plane mm/px varies per patient, Z is 1mm), and unscaled coordinates would
    make an in-plane vessel look different from a through-plane one.

    Returns an array indexed 0..num_features (index 0 is background, unused).
    """
    H, W, Z = labeled_mask.shape
    flat = np.ascontiguousarray(labeled_mask).ravel()
    fg = np.flatnonzero(flat)
    m = num_features + 1
    if fg.size == 0:
        return np.ones(m, dtype=np.float64)

    lz = flat[fg]
    coords = (
        (fg // (W * Z)).astype(np.float64) * spacing[0],
        ((fg // Z) % W).astype(np.float64) * spacing[1],
        (fg % Z).astype(np.float64) * spacing[2],
    )
    cnt = np.bincount(lz, minlength=m).astype(np.float64)
    safe = np.maximum(cnt, 1.0)
    mean = [np.bincount(lz, weights=c, minlength=m) / safe for c in coords]

    cov = np.zeros((m, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i, 3):
            sij = np.bincount(lz, weights=coords[i] * coords[j], minlength=m) / safe
            cij = sij - mean[i] * mean[j]
            cov[:, i, j] = cij
            cov[:, j, i] = cij

    ev = np.linalg.eigvalsh(cov)                      # ascending eigenvalues
    l3 = np.maximum(ev[:, 0], 1e-9)
    l1 = np.maximum(ev[:, 2], 1e-9)
    elong = np.sqrt(l1 / l3)
    elong[cnt < 4] = 1.0        # too few voxels for a meaningful covariance
    return elong


def predict_probs(model, images, device, tta=False):
    """
    Sigmoid probabilities for a batch. With tta=True the prediction is averaged over the
    4 horizontal/vertical flip combinations, each flip undone before accumulating.
    Training used RandFlipd on both spatial axes, so these views are in-distribution.
    """
    views = [()] if not tta else [(), (2,), (3,), (2, 3)]
    acc = None
    for dims in views:
        x = torch.flip(images, dims) if dims else images
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(x)
        else:
            logits = model(x)
        p = torch.sigmoid(logits.float())
        if dims:
            p = torch.flip(p, dims)
        acc = p if acc is None else acc + p
    return acc / len(views)


def remove_small_objects_3d(vol_binary, min_voxels=15, vol_prob=None, min_peak_prob=0.0,
                            max_elongation=0.0, spacing=(1.0, 1.0, 1.0)):
    """
    Applies 3D connected-component labeling on full 3D CT volume (26-connectivity).
    - Removes 3D components with volume < min_voxels.
    - If min_peak_prob > 0 (and vol_prob is supplied): removes 3D components whose PEAK sigmoid
      probability never reaches min_peak_prob.
    - If max_elongation > 0: removes 3D components more elongated than that (tubular vessels),
      measured in millimetres via `spacing`.

    The peak-probability gate is a detection decision applied to whole components, kept separate
    from the pixel binarization threshold, which is a segmentation decision. This lets a low pixel
    threshold preserve nodule boundaries while low-confidence vessel/fissure blobs are deleted
    outright rather than eroded.
    """
    if np.sum(vol_binary) == 0:
        return vol_binary

    use_peak_gate = (min_peak_prob > 0.0 and vol_prob is not None)
    use_shape_gate = (max_elongation > 0.0)

    # Step 1: 3D Connected-Component Size & Peak-Confidence Filtering
    if min_voxels > 0 or use_peak_gate or use_shape_gate:
        labeled_mask, num_features = label(vol_binary, structure=np.ones((3, 3, 3), dtype=bool))
        if num_features == 0:
            return vol_binary
        component_sizes = np.bincount(labeled_mask.ravel(), minlength=num_features + 1)
        too_small = np.zeros(num_features + 1, dtype=bool)

        if min_voxels > 0:
            too_small |= (component_sizes < min_voxels)

        if use_peak_gate:
            # Per-component max over foreground voxels only (labeled_mask is 0 elsewhere)
            flat_labels = labeled_mask.ravel()
            fg = np.flatnonzero(flat_labels)
            comp_peak = np.zeros(num_features + 1, dtype=np.float32)
            np.maximum.at(comp_peak, flat_labels[fg], vol_prob.ravel()[fg].astype(np.float32))
            too_small |= (comp_peak < min_peak_prob)

        if use_shape_gate:
            elong = component_elongation(labeled_mask, num_features, spacing)
            too_small |= (elong > max_elongation)

        too_small[0] = False  # Ensure background is never removed
        cleaned_vol = vol_binary.copy()
        cleaned_vol[too_small[labeled_mask]] = False
    else:
        cleaned_vol = vol_binary.copy()

    return cleaned_vol


def compute_surface_distances(pred_binary, gt_binary, spacing=None):
    """
    Computes HD95 and ASD in millimeters using distance_transform_edt.
    spacing: tuple of mm-per-voxel for each axis, e.g. (mm_y, mm_x) for 2D
             or (mm_y, mm_x, mm_z) for 3D. If None, distances are in pixels.
    """
    if np.sum(pred_binary) == 0 or np.sum(gt_binary) == 0:
        return np.nan, np.nan

    pred_border = pred_binary ^ binary_erosion(pred_binary)
    gt_border = gt_binary ^ binary_erosion(gt_binary)

    if np.sum(pred_border) == 0 or np.sum(gt_border) == 0:
        return np.nan, np.nan

    # EDT from gt/pred borders gives distance of each voxel to nearest surface point
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    d_p2g = dt_gt[pred_border]
    d_g2p = dt_pred[gt_border]

    all_dists = np.concatenate([d_p2g, d_g2p])
    hd95 = float(np.percentile(all_dists, 95))
    asd = float(np.mean(all_dists))
    return hd95, asd


def compute_single_mask_metrics(p_mask, g_mask, spacing=None):
    """
    Computes Dice, IoU, Precision, Sensitivity, Specificity, HD95, ASD, and Failure flag
    with explicit handling for empty masks. HD95/ASD are in true mm when spacing is provided.
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
            "dice": 1.0, "iou": 1.0, "precision": 1.0, "sensitivity": np.nan, "specificity": 1.0,
            "hd95": 0.0, "asd": 0.0, "is_failure": False, "is_false_alarm": False, "is_empty_gt": True
        }

    # Case 2: GT is empty, but Pred contains false positives (False Alarm)
    if g_sum == 0 and p_sum > 0:
        spec = tn / (tn + fp + 1e-8)
        return {
            "dice": 0.0, "iou": 0.0, "precision": 0.0, "sensitivity": np.nan, "specificity": spec,
            "hd95": np.nan, "asd": np.nan, "is_failure": False, "is_false_alarm": True, "is_empty_gt": True
        }

    # Case 3: GT is non-empty, but Pred is empty (False Negative / Missed Tumor)
    if g_sum > 0 and p_sum == 0:
        return {
            "dice": 0.0, "iou": 0.0, "precision": 0.0, "sensitivity": 0.0, "specificity": 1.0,
            "hd95": np.nan, "asd": np.nan, "is_failure": True, "is_false_alarm": False, "is_empty_gt": False
        }

    # Case 4: Both GT and Pred are non-empty
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    hd95, asd = compute_surface_distances(p_mask > 0, g_mask > 0, spacing=spacing)
    is_failure = bool(dice < 0.1)

    return {
        "dice": dice, "iou": iou, "precision": prec, "sensitivity": sens, "specificity": spec,
        "hd95": hd95, "asd": asd, "is_failure": is_failure, "is_false_alarm": False, "is_empty_gt": False
    }


def aggregate_metric_dict(list_of_metric_dicts):
    """
    Summarizes a list of metric dicts into mean values, failure rates, and false alarm rates.
    Filters out np.nan values for sensitivity (empty GT slices).
    """
    if not list_of_metric_dicts:
        return {
            "dice": 0.0, "iou": 0.0, "precision": 0.0, "sensitivity": 0.0, "specificity": 0.0,
            "hd95": 0.0, "asd": 0.0, "failure_rate": 0.0, "false_alarm_rate": 0.0, "count": 0
        }

    dices = [m["dice"] for m in list_of_metric_dicts]
    ious = [m["iou"] for m in list_of_metric_dicts]
    precs = [m["precision"] for m in list_of_metric_dicts]
    senss = [m["sensitivity"] for m in list_of_metric_dicts if not np.isnan(m["sensitivity"])]
    specs = [m["specificity"] for m in list_of_metric_dicts]

    hd95s = [m["hd95"] for m in list_of_metric_dicts if not np.isnan(m["hd95"])]
    asds = [m["asd"] for m in list_of_metric_dicts if not np.isnan(m["asd"])]
    failures = [1.0 if m.get("is_failure", False) else 0.0 for m in list_of_metric_dicts]
    false_alarms = [1.0 if m.get("is_false_alarm", False) else 0.0 for m in list_of_metric_dicts]

    return {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "sensitivity": float(np.mean(senss)) if senss else 0.0,
        "specificity": float(np.mean(specs)),
        "hd95": float(np.mean(hd95s)) if hd95s else 0.0,
        "asd": float(np.mean(asds)) if asds else 0.0,
        "failure_rate": float(np.mean(failures)),
        "false_alarm_rate": float(np.mean(false_alarms)),
        "count": len(list_of_metric_dicts)
    }


def evaluate_test_set_hierarchical(model, loader, device, min_voxels_3d=15,
                                   threshold=0.5, min_peak_prob=0.0,
                                   max_elongation=0.0, tta=False):
    """
    Comprehensive hierarchical evaluation:
    1. Per-Slice evaluation (All slices & Positive tumor slices)
    2. Per-Patient evaluation (Full 3D patient volume reconstruction)
    3. Per-Nodule 3D evaluation & size stratification (Small, Medium, Large nodules)
    """
    model.eval()

    slice_metrics_all = []
    slice_metrics_pos = []

    patient_metrics_all = []
    patient_metrics_pos = []
    per_patient_rows = []

    nodule_metrics_all = []
    nodule_metrics_small = []
    nodule_metrics_medium = []
    nodule_metrics_large = []

    current_pid = None
    current_patient_preds = {}
    current_patient_gts = {}
    current_cropped_shape = (256.0, 256.0)

    def finalize_current_patient():
        nonlocal current_pid, current_patient_preds, current_patient_gts, current_cropped_shape
        if current_pid is None or not current_patient_preds:
            return

        s_keys = sorted(current_patient_preds.keys())
        # current_patient_preds holds raw sigmoid probabilities; binarize once the volume is assembled
        vol_prob = np.stack([current_patient_preds[k] for k in s_keys], axis=-1)
        vol_pred = vol_prob > threshold
        vol_gt = np.stack([current_patient_gts[k] for k in s_keys], axis=-1)

        cs_h, cs_w = current_cropped_shape
        spacing_3d = (cs_h / 256.0, cs_w / 256.0, 1.0)
        spacing_2d = (cs_h / 256.0, cs_w / 256.0)

        # Apply 3D Volumetric Connected-Component Filtering + Peak-Confidence + Shape Gates
        if (min_voxels_3d > 0 or min_peak_prob > 0.0 or max_elongation > 0.0) and np.sum(vol_pred) > 0:
            vol_pred = remove_small_objects_3d(
                vol_pred,
                min_voxels=min_voxels_3d,
                vol_prob=vol_prob,
                min_peak_prob=min_peak_prob,
                max_elongation=max_elongation,
                spacing=spacing_3d
            )

        # 1. 3D Patient Volumetric Metrics
        m_patient = compute_single_mask_metrics(vol_pred, vol_gt, spacing=spacing_3d)
        patient_metrics_all.append(m_patient)
        if not m_patient["is_empty_gt"]:
            patient_metrics_pos.append(m_patient)

        # 2. 2D Per-Slice Metrics from 3D-filtered Volume
        patient_2d_tumor_metrics = []
        patient_2d_all_metrics = []
        has_tumor_scan = (np.sum(vol_gt) > 0)
        num_tumor_slices = 0

        for z_i, k in enumerate(s_keys):
            p_slice = vol_pred[:, :, z_i]
            g_slice = vol_gt[:, :, z_i]

            m_slice = compute_single_mask_metrics(p_slice, g_slice, spacing=spacing_2d)
            slice_metrics_all.append(m_slice)
            patient_2d_all_metrics.append(m_slice)

            if np.sum(g_slice) > 0:
                num_tumor_slices += 1
                slice_metrics_pos.append(m_slice)
                patient_2d_tumor_metrics.append(m_slice)

        agg_tumor_2d = aggregate_metric_dict(patient_2d_tumor_metrics)
        agg_all_2d = aggregate_metric_dict(patient_2d_all_metrics)

        per_patient_rows.append({
            "patient_id": current_pid,
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

        # 3. 3D Nodule Lesion Extraction & Stratification
        if has_tumor_scan:
            labeled_gt, num_nodules = label(vol_gt > 0)
            slices = find_objects(labeled_gt)

            for n_idx, bbox in enumerate(slices, 1):
                if bbox is None:
                    continue
                nodule_gt_mask = (labeled_gt[bbox] == n_idx)
                nodule_pred_mask = vol_pred[bbox]

                vol_voxels = int(np.sum(nodule_gt_mask))
                m_nodule = compute_single_mask_metrics(nodule_pred_mask, nodule_gt_mask, spacing=spacing_3d)
                nodule_metrics_all.append(m_nodule)

                if vol_voxels < 100:
                    nodule_metrics_small.append(m_nodule)
                elif vol_voxels < 1000:
                    nodule_metrics_medium.append(m_nodule)
                else:
                    nodule_metrics_large.append(m_nodule)

        # Clear memory for this patient immediately
        current_patient_preds.clear()
        current_patient_gts.clear()

    filter_parts = [f"≥{min_voxels_3d} voxels"]
    if min_peak_prob > 0.0:
        filter_parts.append(f"peak ≥{min_peak_prob:g}")
    if max_elongation > 0.0:
        filter_parts.append(f"elong ≤{max_elongation:g}")
    if tta:
        filter_parts.append("TTA x4")
    filter_desc = ", ".join(filter_parts)
    print(f"\nRunning Comprehensive Evaluation on 2D Model (3D Filter: {filter_desc}, Threshold: {threshold})...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Evaluating Slices]"):
            images, masks, pids, s_idxs = batch[0], batch[1], batch[2], batch[3]
            cropped_shapes = batch[4]  # list of [H, W, Z] per sample
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Keep raw probabilities: the peak-confidence gate needs them at component level,
            # and binarization is deferred until the 3D volume is assembled per patient.
            probs_np = predict_probs(model, images, device, tta=tta).cpu().numpy()
            masks_np = masks.cpu().numpy()

            for b in range(probs_np.shape[0]):
                p_prob = probs_np[b, 0].astype(np.float32)
                g_mask = (masks_np[b, 0] > 0.5)
                pid = str(pids[b])
                s_idx = int(s_idxs[b]) if not isinstance(s_idxs[b], int) else s_idxs[b]

                cs_h = float(cropped_shapes[0][b])
                cs_w = float(cropped_shapes[1][b])

                if pid != current_pid:
                    finalize_current_patient()
                    current_pid = pid
                    current_cropped_shape = (cs_h, cs_w)

                current_patient_preds[s_idx] = p_prob
                current_patient_gts[s_idx] = g_mask

    # Finalize the last patient after loop ends
    finalize_current_patient()

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


def print_and_format_report(results, min_voxels_3d, report_path,
                            threshold=0.5, min_peak_prob=0.0,
                            max_elongation=0.0, tta=False):
    """
    Formats clean metric tables and saves full report to file.
    """
    filter_parts = [f"≥{min_voxels_3d} voxels"]
    if min_peak_prob > 0.0:
        filter_parts.append(f"peak ≥{min_peak_prob:g}")
    if max_elongation > 0.0:
        filter_parts.append(f"elong ≤{max_elongation:g}")
    if tta:
        filter_parts.append("TTA x4")
    filter_desc = ", ".join(filter_parts)
    lines = []
    lines.append("=========================================================================================================")
    lines.append(f"                   HIERARCHICAL EVALUATION REPORT (3D Filter: {filter_desc}, Threshold: {threshold})")
    lines.append("=========================================================================================================\n")

    def format_row(title, m):
        return (
            f"  {title:<32s} | Dice: {m['dice']:.4f} | IoU: {m['iou']:.4f} | "
            f"Prec: {m['precision']:.4f} | Sens: {m['sensitivity']:.4f} | Spec: {m['specificity']:.4f} | "
            f"HD95: {m['hd95']:5.2f}mm | ASD: {m['asd']:5.2f}mm | Fail: {m['failure_rate']*100:5.1f}% | "
            f"FA: {m.get('false_alarm_rate', 0.0)*100:5.1f}% | Count: {m['count']:4d}"
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


def load_trained_model(model_path, device, in_channels=1):
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint file '{model_path}' not found.")
        sys.exit(1)

    print(f"Loading MONAI model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if "model_type" in checkpoint:
        try:
            model, _ = get_model(checkpoint["model_type"], in_channels=in_channels)
            model.load_state_dict(state_dict)
            model.eval()
            return model.to(device)
        except Exception:
            pass

    import inspect
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
            model, _ = get_model(m_type, in_channels=in_channels)
            model.load_state_dict(state_dict)
            model.eval()
            return model.to(device)
        except Exception:
            continue

    raise RuntimeError(f"Failed to load checkpoint from {model_path} into any supported architecture.")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hierarchical Evaluation of MONAI 2D UNet")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help=f"Path to manifest CSV (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to trained model checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default=DEFAULT_SPLIT, help=f"Dataset split to evaluate: train, val, or test (default: {DEFAULT_SPLIT})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help=f"DataLoader num_workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--min_voxels_3d", "--min_voxels", "--min_size", type=int, default=DEFAULT_MIN_VOXELS_3D, help=f"Minimum 3D connected component volume in voxels (default: {DEFAULT_MIN_VOXELS_3D})")
    parser.add_argument("--min_peak_prob", "--peak_gate", type=float, default=DEFAULT_MIN_PEAK_PROB, help=f"Keep a 3D connected component only if its peak sigmoid probability reaches this value. Applied to whole components, so it deletes low-confidence blobs without eroding nodule boundaries. 0 disables the gate (default: {DEFAULT_MIN_PEAK_PROB})")
    parser.add_argument("--max_elongation", "--max_elong", type=float, default=DEFAULT_MAX_ELONGATION, help=f"Drop 3D components whose sqrt(lambda1/lambda3) exceeds this, i.e. tubular vessels. Nodules sit near 1; ~2.5-3.0 is a reasonable cut. 0 disables (default: {DEFAULT_MAX_ELONGATION})")
    parser.add_argument("--tta", type=int, default=DEFAULT_TTA, help=f"1 = average the sigmoid over 4 horizontal/vertical flips. Costs ~4x forward passes but the pipeline is I/O bound, so wall-clock impact is small (default: {DEFAULT_TTA})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Probability threshold for binarizing predictions (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--report_path", type=str, default=DEFAULT_REPORT_PATH, help=f"Path for evaluation text report (default: {DEFAULT_REPORT_PATH})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Global random seed for full evaluation reproducibility (default: {DEFAULT_SEED})")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args.model_path, device, in_channels=1)

    _, val_transforms = get_transforms()
    eval_dataset = LIDC2DDataset(args.manifest, split=args.split, transform=val_transforms, seed=args.seed)
    eval_dataset.data_entries = eval_dataset.full_split_df
    print(f"Loaded {len(eval_dataset)} '{args.split}' slices from {args.manifest}.")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    results = evaluate_test_set_hierarchical(
        model,
        eval_loader,
        device,
        min_voxels_3d=args.min_voxels_3d,
        threshold=args.threshold,
        min_peak_prob=args.min_peak_prob,
        max_elongation=args.max_elongation,
        tta=bool(args.tta)
    )
    print_and_format_report(
        results,
        min_voxels_3d=args.min_voxels_3d,
        report_path=args.report_path,
        threshold=args.threshold,
        min_peak_prob=args.min_peak_prob,
        max_elongation=args.max_elongation,
        tta=bool(args.tta)
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
