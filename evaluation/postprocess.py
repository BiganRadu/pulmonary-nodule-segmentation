"""Post-processing shared by evaluation and by the training-time validation pass.

Kept in one module so the gate applied while a model trains is byte-identical to the one
that produces the reported metrics -- a copy in each file would drift the moment either
side was tuned.

Stage order (all applied to the reconstructed 3D volume, 26-connectivity):
    1. probability threshold          -> binary volume
    2. min_voxels                     -> drop specks
    3. min_peak_prob                  -> drop components that never reach a confidence
    4. max_elongation                 -> drop tubular components (vessels), in millimetres
"""
import numpy as np
from scipy.ndimage import label


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

        comp_peak = None
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
