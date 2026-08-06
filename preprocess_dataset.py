# Standard library
import os
import sys
import glob
import math
import json
import argparse
import xml.etree.ElementTree as ET
import time
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

# 3rd party
import cv2
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, binary_fill_holes, label

# Default Configuration Constants
DEFAULT_DATASET_DIR = "/mnt/hdd/CS2023-2027/AIMAS/practica/datasetmare" if os.path.exists("/mnt/hdd/CS2023-2027/AIMAS/practica/datasetmare") else r"E:\CS2023-2027\AIMAS\practica\datasetmare"
DEFAULT_OUTPUT_DIR = "preprocessed_data2"
DEFAULT_NUM_WORKERS = 6
DEFAULT_NEG_POS_RATIO = 1.5
DEFAULT_MAX_NEG_SLICES = 20
DEFAULT_CONSENSUS_RATIO = 0.5  # 50% strict majority threshold
DEFAULT_TARGET_SPACING = (1.0, 1.0, 1.0)  # Isotropic 1.0mm x 1.0mm x 1.0mm voxel resolution
DEFAULT_CROP_PADDING = 10
DEFAULT_LUNG_Z_TRIM_RATIO = 0.25


def build_xml_index(dataset_dir):
    """
    Indexes XML files in dataset by StudyInstanceUID, SeriesInstanceUID, and imageSOP_UIDs.
    """
    xml_dir = os.path.join(dataset_dir, "tcia-lidc-xml")
    if not os.path.exists(xml_dir):
        xml_dir = dataset_dir

    xml_files = glob.glob(os.path.join(xml_dir, "**/*.xml"), recursive=True)
    single_xml = os.path.join(dataset_dir, "161-resubmitted-correction-3-9-12.xml")
    if os.path.exists(single_xml) and single_xml not in xml_files:
        xml_files.append(single_xml)

    index = {'study': {}, 'series': {}, 'sop': {}, 'patient': {}}
    ns = {'nih': 'http://www.nih.gov'}

    print(f"Indexing {len(xml_files)} XML annotation files...")
    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            header = root.find('nih:ResponseHeader', ns)
            if header is None:
                header = root

            study_uid = (header.findtext('nih:StudyInstanceUID', default='', namespaces=ns) or header.findtext('nih:StudyInstanceUid', default='', namespaces=ns)).strip()
            series_uid = (header.findtext('nih:SeriesInstanceUid', default='', namespaces=ns) or header.findtext('nih:SeriesInstanceUID', default='', namespaces=ns)).strip()

            fname = os.path.basename(xf)
            if "LIDC-IDRI-" in fname:
                pid_part = fname.split(".")[0]
                index['patient'][pid_part] = xf

            if study_uid:
                index['study'][study_uid] = xf
            if series_uid:
                index['series'][series_uid] = xf

            for sop_elem in root.findall('.//nih:imageSOP_UID', ns):
                if sop_elem.text:
                    sop_clean = sop_elem.text.strip()
                    index['sop'][sop_clean] = xf
        except Exception as e:
            print(f"[Warning] Failed to parse XML file {xf}: {e}")
            continue

    print(f"XML Index ready: {len(index['study'])} studies, {len(index['series'])} series, {len(index['sop'])} SOPs.")
    return index


def get_xml_target_uids(xml_file):
    """
    Extracts StudyInstanceUID and SeriesInstanceUID from an XML annotation file.
    """
    if not xml_file or not os.path.exists(xml_file):
        return None, None

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        ns = {'nih': 'http://www.nih.gov'}
        header = root.find('nih:ResponseHeader', ns)
        if header is None:
            header = root

        study_uid = (header.findtext('nih:StudyInstanceUID', default='', namespaces=ns) or header.findtext('nih:StudyInstanceUid', default='', namespaces=ns)).strip()
        series_uid = (header.findtext('nih:SeriesInstanceUid', default='', namespaces=ns) or header.findtext('nih:SeriesInstanceUID', default='', namespaces=ns)).strip()
        return study_uid, series_uid
    except Exception as e:
        print(f"[Warning] Error reading XML header from {xml_file}: {e}")
        return None, None


def load_volumetric_dicom(patient_id, dataset_dir, xml_index=None):
    """
    Pipeline Step 1: Standard Volumetric DICOM Conversion
    Loads raw DICOM volume for a patient and extracts full 3D spatial metadata.
    """
    lidc_dir = os.path.join(dataset_dir, "lidc_idri")

    clean_id = os.path.basename(patient_id.strip('/\\'))
    if not clean_id.startswith("LIDC-IDRI-"):
        if clean_id.isdigit():
            clean_id = f"LIDC-IDRI-{int(clean_id):04d}"
        else:
            clean_id = f"LIDC-IDRI-{clean_id}"

    patient_path = os.path.join(lidc_dir, clean_id)
    if not os.path.exists(patient_path):
        matches = glob.glob(os.path.join(lidc_dir, f"*{clean_id}*"))
        if matches:
            patient_path = matches[0]
            clean_id = os.path.basename(patient_path)
        else:
            raise FileNotFoundError(f"Could not find patient directory for '{patient_id}' in '{lidc_dir}'")

    dcm_files = glob.glob(os.path.join(patient_path, "**/*.dcm"), recursive=True)
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {patient_path}")

    # Find matching XML file first
    matched_xml = None
    if xml_index:
        if clean_id in xml_index['patient']:
            matched_xml = xml_index['patient'][clean_id]
        else:
            xml_in_folder = glob.glob(os.path.join(patient_path, "*.xml"))
            if xml_in_folder:
                matched_xml = xml_in_folder[0]

    target_study_uid, target_series_uid = get_xml_target_uids(matched_xml)

    # Group DICOM files by SeriesInstanceUID
    series_groups = {}
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            series_uid = str(getattr(ds, 'SeriesInstanceUID', 'default_series'))
            modality = str(getattr(ds, 'Modality', 'CT'))
            if modality != 'CT':
                continue
            if series_uid not in series_groups:
                series_groups[series_uid] = []

            z_pos = float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else float(getattr(ds, 'SliceLocation', 0))
            sop_uid = str(getattr(ds, 'SOPInstanceUID', ''))
            study_uid = str(getattr(ds, 'StudyInstanceUID', ''))
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            rows = int(getattr(ds, 'Rows', 512))
            cols = int(getattr(ds, 'Columns', 512))

            pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
            row_spacing, col_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])

            img_pos = [float(val) for val in getattr(ds, 'ImagePositionPatient', [0.0, 0.0, z_pos])]
            img_orient = [float(val) for val in getattr(ds, 'ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])]

            series_groups[series_uid].append({
                'path': f,
                'z': z_pos,
                'sop': sop_uid,
                'study_uid': study_uid,
                'series_uid': series_uid,
                'slope': slope,
                'intercept': intercept,
                'rows': rows,
                'cols': cols,
                'row_spacing': row_spacing,
                'col_spacing': col_spacing,
                'img_pos': img_pos,
                'img_orient': img_orient
            })
        except Exception as e:
            print(f"[Warning] Could not read DICOM file {f}: {e}")
            continue

    if not series_groups:
        raise ValueError(f"No valid CT series found in {patient_path}")

    # Lookup via SOP if not matched yet
    if not matched_xml and xml_index:
        for s_uid, s_list in series_groups.items():
            for item in s_list:
                if item['sop'] in xml_index['sop']:
                    matched_xml = xml_index['sop'][item['sop']]
                    target_study_uid, target_series_uid = get_xml_target_uids(matched_xml)
                    break
            if matched_xml:
                break

    if not matched_xml or not os.path.exists(matched_xml):
        raise FileNotFoundError(f"Missing or invalid XML annotation file for patient '{clean_id}'")

    # Strict CT Series selection matching XML SeriesInstanceUID
    if target_series_uid and target_series_uid in series_groups:
        selected_series_uid = target_series_uid
    else:
        raise ValueError(
            f"Target XML SeriesInstanceUID '{target_series_uid}' not found in DICOM CT series for patient '{clean_id}'. "
            f"Available series UIDs: {list(series_groups.keys())}"
        )

    slices_info = series_groups[selected_series_uid]
    slices_info.sort(key=lambda s: s['z'])

    rows, cols = slices_info[0]['rows'], slices_info[0]['cols']
    num_z = len(slices_info)

    # Compute Z-slice thickness from physical slice positions
    if num_z > 1:
        z_spacing = abs(slices_info[1]['z'] - slices_info[0]['z'])
        if z_spacing == 0.0:
            z_spacing = 1.25
    else:
        z_spacing = 1.25

    row_spacing = slices_info[0]['row_spacing']
    col_spacing = slices_info[0]['col_spacing']

    volume = np.zeros((rows, cols, num_z), dtype=np.float32)
    for z_idx, s in enumerate(slices_info):
        ds = pydicom.dcmread(s['path'])
        pixel_array = ds.pixel_array.astype(np.float32)
        hu_pixels = pixel_array * s['slope'] + s['intercept']
        volume[:, :, z_idx] = hu_pixels

    spatial_meta = {
        'series_instance_uid': selected_series_uid,
        'original_spacing': [row_spacing, col_spacing, z_spacing],
        'original_shape': [rows, cols, num_z],
        'origin': slices_info[0]['img_pos'],
        'orientation': slices_info[0]['img_orient'],
        'z_direction_flipped': False,
        'y_direction_flipped': False,
        'x_direction_flipped': False
    }

    return volume, slices_info, matched_xml, clean_id, spatial_meta


def compute_pairwise_dice(mask_a, mask_b):
    """
    Computes 3D Dice Similarity Coefficient between two binary masks.
    """
    intersection = np.sum((mask_a > 0) & (mask_b > 0))
    sum_a = np.sum(mask_a > 0)
    sum_b = np.sum(mask_b > 0)

    if sum_a == 0 and sum_b == 0:
        return 1.0
    if sum_a + sum_b == 0:
        return 0.0
    return float(2.0 * intersection / (sum_a + sum_b))


def parse_annotations_and_consensus(xml_file, slices_info, consensus_ratio=DEFAULT_CONSENSUS_RATIO):
    """
    Parses LIDC XML expert annotations:
    - Preserves individual 3D binary masks per radiologist session
    - Computes strict majority consensus mask (e.g. >= 2/3 or >= 2/4 radiologists)
    - Calculates inter-annotator agreement metrics (mean pairwise Dice)
    """
    rows, cols = slices_info[0]['rows'], slices_info[0]['cols']
    num_z = len(slices_info)

    if not xml_file or not os.path.exists(xml_file):
        raise FileNotFoundError(f"XML file '{xml_file}' does not exist.")

    sop_to_zidx = {s['sop']: i for i, s in enumerate(slices_info)}

    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'nih': 'http://www.nih.gov'}

    sessions = root.findall('nih:readingSession', ns)
    if not sessions:
        raise ValueError(f"No readingSession annotations found in XML file '{xml_file}'.")

    num_sessions = len(sessions)
    session_masks = np.zeros((num_sessions, rows, cols, num_z), dtype=np.uint8)
    consensus_count = np.zeros((rows, cols, num_z), dtype=np.uint8)

    for sess_idx, s in enumerate(sessions):
        sess_mask = np.zeros((rows, cols, num_z), dtype=np.uint8)
        nodules = s.findall('nih:unblindedReadNodule', ns)

        for nod in nodules:
            for roi in nod.findall('nih:roi', ns):
                sop = roi.findtext('nih:imageSOP_UID', default='', namespaces=ns).strip()
                if sop in sop_to_zidx:
                    z_idx = sop_to_zidx[sop]
                    edges = roi.findall('nih:edgeMap', ns)
                    if len(edges) >= 3:
                        poly = np.array([[(int(e.findtext('nih:xCoord', namespaces=ns)), int(e.findtext('nih:yCoord', namespaces=ns))) for e in edges]], dtype=np.int32)
                        slice_2d = np.ascontiguousarray(sess_mask[:, :, z_idx])
                        cv2.fillPoly(slice_2d, poly, 1)
                        sess_mask[:, :, z_idx] = slice_2d

        session_masks[sess_idx] = sess_mask
        consensus_count += (sess_mask > 0).astype(np.uint8)

    if num_sessions == 1:
        min_agreement = 1
    else:
        min_agreement = max(2, math.ceil(num_sessions * consensus_ratio))

    consensus_mask = (consensus_count >= min_agreement).astype(np.uint8)

    pairwise_dices = []
    if num_sessions > 1:
        for i in range(num_sessions):
            for j in range(i + 1, num_sessions):
                dice_val = compute_pairwise_dice(session_masks[i], session_masks[j])
                pairwise_dices.append(dice_val)
        mean_inter_dice = round(float(np.mean(pairwise_dices)), 4)
    else:
        mean_inter_dice = 1.0

    return consensus_mask, session_masks, num_sessions, mean_inter_dice


def canonical_reorient(volume, consensus_mask, session_masks, spatial_meta):
    """
    Pipeline Step 2: Canonical Anatomical Orientation (RAS+ Alignment)
    Re-orients CT volume and binary masks so that axes consistently follow:
    Axis 0: Right -> Left
    Axis 1: Posterior -> Anterior
    Axis 2: Inferior -> Superior (Cap-up)
    """
    orient = spatial_meta['orientation']
    # Check direction cosines
    x_dir = orient[0]  # Row direction X
    y_dir = orient[4]  # Col direction Y

    vol_re = volume
    con_re = consensus_mask
    ses_re = session_masks

    # Flip X if inverted
    if x_dir < 0:
        vol_re = np.flip(vol_re, axis=0)
        con_re = np.flip(con_re, axis=0)
        ses_re = np.flip(ses_re, axis=1)  # 4D: (N, H, W, Z)
        spatial_meta['x_direction_flipped'] = True

    # Flip Y if inverted
    if y_dir < 0:
        vol_re = np.flip(vol_re, axis=1)
        con_re = np.flip(con_re, axis=1)
        ses_re = np.flip(ses_re, axis=2)
        spatial_meta['y_direction_flipped'] = True

    return vol_re, con_re, ses_re, spatial_meta


def resample_volume_and_masks(volume, consensus_mask, session_masks, spatial_meta, target_spacing=DEFAULT_TARGET_SPACING):
    """
    Pipeline Step 3: Isotropic / Uniform Spacing Resampling
    Resamples 3D CT volume (order=1 linear) and binary masks (order=0 nearest) to target (1.0mm, 1.0mm, 1.0mm).
    """
    orig_sp = spatial_meta['original_spacing']  # [row_sp, col_sp, z_sp]
    zoom_factors = [orig_sp[0] / target_spacing[0], orig_sp[1] / target_spacing[1], orig_sp[2] / target_spacing[2]]

    # Resample CT HU volume (order=1 spline/linear)
    res_vol = zoom(volume, zoom_factors, order=1, mode='nearest').astype(np.float32)

    # Resample consensus mask (order=0 nearest neighbor)
    res_con = zoom(consensus_mask, zoom_factors, order=0, mode='nearest').astype(np.uint8)

    # Resample 4D session masks: (N_sessions, H, W, Z)
    num_sessions = session_masks.shape[0]
    res_ses_list = []
    for s_idx in range(num_sessions):
        s_res = zoom(session_masks[s_idx], zoom_factors, order=0, mode='nearest').astype(np.uint8)
        res_ses_list.append(s_res)
    res_ses = np.stack(res_ses_list, axis=0)

    spatial_meta['resampled_spacing'] = list(target_spacing)
    spatial_meta['zoom_factors'] = zoom_factors
    spatial_meta['resampled_shape'] = list(res_vol.shape)

    return res_vol, res_con, res_ses, spatial_meta


def apply_hu_windowing(hu_volume, min_hu=-1000.0, max_hu=400.0):
    """
    Pipeline Step 4: Configurable HU Windowing & Normalization
    Clips CT volume to [-1000, 400] HU lung window and normalizes to [0.0, 1.0].
    """
    clipped = np.clip(hu_volume, min_hu, max_hu)
    normalized = (clipped - min_hu) / (max_hu - min_hu)
    return normalized.astype(np.float32)


def crop_lung_cavity(norm_volume, hu_volume, consensus_mask, session_masks, spatial_meta, padding, lung_z_trim_ratio):
    """
    Pipeline Step 5: Anatomical Lung Cavity Cropping
    1. Builds a 2D body mask per axial slice (fill_holes works correctly on closed 2D rings).
    2. Extracts internal lung air (HU in [-1000, -400]) inside the body.
    3. Trims Z-slices whose internal lung air count < lung_z_trim_ratio * peak slice count.
       Tumor-containing slices are never trimmed.
    4. Crops Y/X to the bounding box of the remaining lung air + consensus mask.
    """
    H, W, Z = norm_volume.shape

    # --- 2D slice-by-slice body mask (fixes the 3D fill_holes border-leak bug) ---
    body_mask = np.zeros((H, W, Z), dtype=bool)
    for z in range(Z):
        body_mask[:, :, z] = binary_fill_holes(hu_volume[:, :, z] > -500.0)

    lung_air = ((hu_volume >= -1000.0) & (hu_volume <= -400.0)) & body_mask

    # --- Z-slice trimming based on internal lung air ---
    air_per_z = np.sum(lung_air, axis=(0, 1))          # shape (Z,)
    tumor_per_z = np.sum(consensus_mask > 0, axis=(0, 1))
    peak_air = air_per_z.max() if air_per_z.max() > 0 else 1
    min_air = peak_air * lung_z_trim_ratio

    valid_z = np.where((air_per_z >= min_air) | (tumor_per_z > 0))[0]

    if len(valid_z) == 0:
        crop_bbox = [0, H, 0, W, 0, Z]
    else:
        z_min = max(0, int(valid_z[0]) - padding)
        z_max = min(Z, int(valid_z[-1]) + 1 + padding)

        # Y/X bounds within the valid Z range
        sub_mask = lung_air[:, :, z_min:z_max] | (consensus_mask[:, :, z_min:z_max] > 0)
        nz = np.argwhere(sub_mask)
        if len(nz) == 0:
            y_min, y_max, x_min, x_max = 0, H, 0, W
        else:
            y_min = max(0, int(nz[:, 0].min()) - padding)
            y_max = min(H, int(nz[:, 0].max()) + 1 + padding)
            x_min = max(0, int(nz[:, 1].min()) - padding)
            x_max = min(W, int(nz[:, 1].max()) + 1 + padding)

        crop_bbox = [y_min, y_max, x_min, x_max, z_min, z_max]

    y_min, y_max, x_min, x_max, z_min, z_max = crop_bbox
    crop_vol = norm_volume[y_min:y_max, x_min:x_max, z_min:z_max]
    crop_con = consensus_mask[y_min:y_max, x_min:x_max, z_min:z_max]
    crop_ses = session_masks[:, y_min:y_max, x_min:x_max, z_min:z_max]

    spatial_meta['crop_bbox'] = crop_bbox
    spatial_meta['cropped_shape'] = list(crop_vol.shape)

    return crop_vol, crop_con, crop_ses, spatial_meta


def audit_nodule_volume(orig_consensus_mask, resampled_consensus_mask, orig_spacing, target_spacing):
    """
    Pipeline Step 8: Audit of Volume Modifications After Resampling
    Tracks physical nodule volume in mm^3 before vs after resampling.
    """
    voxel_vol_orig = orig_spacing[0] * orig_spacing[1] * orig_spacing[2]
    voxel_vol_res = target_spacing[0] * target_spacing[1] * target_spacing[2]

    v_orig_mm3 = round(float(np.sum(orig_consensus_mask > 0) * voxel_vol_orig), 2)
    v_res_mm3 = round(float(np.sum(resampled_consensus_mask > 0) * voxel_vol_res), 2)

    if v_orig_mm3 > 0:
        v_ratio = round(v_res_mm3 / v_orig_mm3, 4)
    else:
        v_ratio = 1.0

    return v_orig_mm3, v_res_mm3, v_ratio


def reverse_transform_mask(cropped_resampled_mask, spatial_meta):
    """
    Pipeline Step 7: Inverse Geometry Mask Reconstruction
    Un-crops and un-resamples a binary mask back to exact original DICOM coordinates (512, 512, Z_orig).
    """
    resamp_shape = spatial_meta['resampled_shape']
    crop_bbox = spatial_meta['crop_bbox']
    zoom_factors = spatial_meta['zoom_factors']

    # Step 1: Un-crop back to resampled volume shape
    uncropped_mask = np.zeros(resamp_shape, dtype=np.uint8)
    y_min, y_max, x_min, x_max, z_min, z_max = crop_bbox
    uncropped_mask[y_min:y_max, x_min:x_max, z_min:z_max] = cropped_resampled_mask

    # Step 2: Un-resample back to original volume shape
    inv_zoom_factors = [1.0 / zf for zf in zoom_factors]
    unresampled_mask = zoom(uncropped_mask, inv_zoom_factors, order=0, mode='nearest').astype(np.uint8)

    # Pad or crop to exact original shape if rounding differences occur
    orig_shape = spatial_meta['original_shape']
    final_mask = np.zeros(orig_shape, dtype=np.uint8)

    h_m = min(orig_shape[0], unresampled_mask.shape[0])
    w_m = min(orig_shape[1], unresampled_mask.shape[1])
    z_m = min(orig_shape[2], unresampled_mask.shape[2])

    final_mask[:h_m, :w_m, :z_m] = unresampled_mask[:h_m, :w_m, :z_m]

    # Step 3: Undo canonical orientation flips if applied
    if spatial_meta.get('y_direction_flipped', False):
        final_mask = np.flip(final_mask, axis=1)
    if spatial_meta.get('x_direction_flipped', False):
        final_mask = np.flip(final_mask, axis=0)

    return final_mask


def extract_nodule_and_xml_audit(matched_xml, ses_masks, con_mask, spatial_meta, num_sessions, clean_id):
    """
    Extracts detailed audit metrics per patient and series:
    - Number of slices, voxel spacing, orientation, dimensions
    - Number of radiologists, number of nodules
    - Nodules annotated by 1, 2, 3, or 4 radiologists
    - Individual nodule volumes in mm^3
    - XML & Series status
    - Malignancy score distribution
    """
    xml_status = "OK" if (matched_xml and os.path.exists(matched_xml)) else "Missing/Invalid"
    series_status = "OK"

    sp = spatial_meta['original_spacing']
    spacing_str = f"{sp[2]:.3f}x{sp[0]:.3f}x{sp[1]:.3f}mm"
    dims = spatial_meta['original_shape']
    dimensions_str = f"{dims[0]}x{dims[1]}x{dims[2]}"
    orientation_str = str(spatial_meta.get('orientation', []))

    if xml_status != "OK" or ses_masks is None or len(ses_masks) == 0:
        return {
            'patient_id': clean_id,
            'series_instance_uid': spatial_meta.get('series_instance_uid', 'N/A'),
            'num_slices': dims[2],
            'spacing': spacing_str,
            'orientation': orientation_str,
            'dimensions': dimensions_str,
            'num_radiologists': num_sessions or 0,
            'num_nodules': 0,
            'nodules_by_1_rad': 0,
            'nodules_by_2_rad': 0,
            'nodules_by_3_rad': 0,
            'nodules_by_4_rad': 0,
            'nodule_volumes_mm3': "[]",
            'xml_status': xml_status,
            'series_status': series_status,
            'malignancy_distribution': "{1: 0, 2: 0, 3: 0, 4: 0, 5: 0}"
        }

    # 3D connected component analysis for physical nodules across radiologist sessions
    union_mask = (np.sum(ses_masks, axis=0) > 0).astype(np.uint8)
    labeled_nodules, num_nodules = label(union_mask)

    voxel_vol = sp[0] * sp[1] * sp[2]
    rads_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    volumes = []

    for nod_idx in range(1, num_nodules + 1):
        nod_mask = (labeled_nodules == nod_idx)
        n_rads = sum(1 for s in range(num_sessions) if np.any(ses_masks[s] & nod_mask))
        if n_rads in rads_counts:
            rads_counts[n_rads] += 1
        else:
            rads_counts[n_rads] = 1
        vol_mm3 = float(np.sum(con_mask & nod_mask) * voxel_vol)
        volumes.append(round(vol_mm3, 2))

    # Extract malignancy distribution from XML
    mal_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    try:
        tree = ET.parse(matched_xml)
        root = tree.getroot()
        ns = {'nih': 'http://www.nih.gov'}
        for s in root.findall('nih:readingSession', ns):
            for nod in s.findall('nih:unblindedReadNodule', ns):
                mal_str = nod.findtext('nih:characteristics/nih:malignancy', default='', namespaces=ns).strip()
                if mal_str.isdigit():
                    val = int(mal_str)
                    if 1 <= val <= 5:
                        mal_counts[val] += 1
    except Exception:
        pass

    return {
        'patient_id': clean_id,
        'series_instance_uid': spatial_meta.get('series_instance_uid', 'N/A'),
        'num_slices': dims[2],
        'spacing': spacing_str,
        'orientation': orientation_str,
        'dimensions': dimensions_str,
        'num_radiologists': num_sessions,
        'num_nodules': num_nodules,
        'nodules_by_1_rad': rads_counts.get(1, 0),
        'nodules_by_2_rad': rads_counts.get(2, 0),
        'nodules_by_3_rad': rads_counts.get(3, 0),
        'nodules_by_4_rad': rads_counts.get(4, 0),
        'nodule_volumes_mm3': str(volumes),
        'xml_status': xml_status,
        'series_status': series_status,
        'malignancy_distribution': str(mal_counts)
    }


def process_single_patient(
    patient_id,
    dataset_dir=DEFAULT_DATASET_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    xml_index=None,
    neg_pos_ratio=DEFAULT_NEG_POS_RATIO,
    max_neg_slices=DEFAULT_MAX_NEG_SLICES,
    crop_padding=DEFAULT_CROP_PADDING,
    lung_z_trim_ratio=DEFAULT_LUNG_Z_TRIM_RATIO,
    keep_all_slices=False,
    return_preview_data=False
):
    """
    Executes the Complete 9-Step Medically Correct DICOM Preprocessing Pipeline for a Single Patient.
    """
    # Step 1: Standard Volumetric DICOM Conversion
    volume, slices_info, matched_xml, clean_id, spatial_meta = load_volumetric_dicom(patient_id, dataset_dir, xml_index)

    # Parse XML annotations & compute strict majority consensus
    orig_consensus_mask, orig_session_masks, num_radiologists, inter_annotator_dice = parse_annotations_and_consensus(matched_xml, slices_info)

    # Step 2: Canonical Anatomical Orientation (RAS+ Alignment)
    vol_can, con_can, ses_can, spatial_meta = canonical_reorient(volume, orig_consensus_mask, orig_session_masks, spatial_meta)
    del volume

    # Step 3: Isotropic / Uniform Spacing Resampling (1.0mm x 1.0mm x 1.0mm)
    res_vol, res_con, res_ses, spatial_meta = resample_volume_and_masks(vol_can, con_can, ses_can, spatial_meta)
    del vol_can, con_can, ses_can

    # Step 8: Audit Nodule Volume Modifications After Resampling
    v_orig_mm3, v_res_mm3, v_ratio = audit_nodule_volume(orig_consensus_mask, res_con, spatial_meta['original_spacing'], DEFAULT_TARGET_SPACING)

    # Step 4: Configurable HU Windowing & Normalization
    norm_vol = apply_hu_windowing(res_vol)

    # Step 5: Anatomical Lung Cavity Cropping
    crop_vol, crop_con, crop_ses, spatial_meta = crop_lung_cavity(
        norm_vol, res_vol, res_con, res_ses, spatial_meta, crop_padding, lung_z_trim_ratio
    )
    del res_vol, norm_vol

    # Verify Inverse Geometry Mask Reconstruction (Step 7 validation)
    recon_mask = reverse_transform_mask(crop_con, spatial_meta)
    recon_dice = compute_pairwise_dice(orig_consensus_mask, recon_mask)
    del recon_mask

    num_slices = crop_vol.shape[2]
    positive_slices = []
    negative_slices = []

    for z in range(num_slices):
        slice_img = crop_vol[:, :, z]  # 2D cropped normalized CT
        slice_mask = crop_con[:, :, z]  # 2D cropped consensus mask
        slice_sessions = crop_ses[:, :, :, z]  # 2D cropped session masks

        tumor_pixels = int(np.sum(slice_mask > 0))

        record = {
            'patient_id': clean_id,
            'slice_idx': z,
            'has_tumor': 1 if tumor_pixels > 0 else 0,
            'tumor_pixels': tumor_pixels,
            'image_slice': slice_img,
            'mask_slice': slice_mask,
            'session_slices': slice_sessions
        }

        if tumor_pixels > 0:
            positive_slices.append(record)
        else:
            negative_slices.append(record)

    # Sample negative slices (or keep all for val/test splits)
    if keep_all_slices:
        sampled_negatives = negative_slices
    else:
        num_pos = len(positive_slices)
        if num_pos > 0:
            num_neg_to_keep = min(len(negative_slices), int(num_pos * neg_pos_ratio) + 2)
        else:
            num_neg_to_keep = min(len(negative_slices), max_neg_slices)

        if len(negative_slices) > num_neg_to_keep:
            indices = np.linspace(0, len(negative_slices) - 1, num_neg_to_keep, dtype=int)
            sampled_negatives = [negative_slices[i] for i in indices]
        else:
            sampled_negatives = negative_slices

    selected_slices = positive_slices + sampled_negatives
    selected_slices.sort(key=lambda r: r['slice_idx'])

    # Save to .npz files with individual radiologist session masks & spatial metadata
    patient_out_dir = os.path.join(output_dir, clean_id)
    os.makedirs(patient_out_dir, exist_ok=True)

    spatial_meta_json = json.dumps(spatial_meta)
    manifest_entries = []

    for r in selected_slices:
        filename = f"{clean_id}_slice{r['slice_idx']:03d}.npz"
        filepath = os.path.join(patient_out_dir, filename).replace("\\", "/")

        img_tensor = np.expand_dims(r['image_slice'], axis=0).astype(np.float32)
        mask_tensor = np.expand_dims(r['mask_slice'], axis=0).astype(np.uint8)
        session_tensors = r['session_slices'].astype(np.uint8)

        np.savez_compressed(
            filepath,
            image=img_tensor,
            mask=mask_tensor,
            session_masks=session_tensors,
            spatial_meta=spatial_meta_json
        )

        manifest_entries.append({
            'patient_id': clean_id,
            'slice_idx': r['slice_idx'],
            'has_tumor': r['has_tumor'],
            'tumor_pixels': r['tumor_pixels'],
            'num_radiologists': num_radiologists,
            'inter_annotator_dice': inter_annotator_dice,
            'v_orig_mm3': v_orig_mm3,
            'v_resamp_mm3': v_res_mm3,
            'v_retention_ratio': v_ratio,
            'recon_dice': recon_dice,
            'filepath': filepath
        })

    audit_record = extract_nodule_and_xml_audit(matched_xml, orig_session_masks, orig_consensus_mask, spatial_meta, num_radiologists, clean_id)

    summary_info = {
        'num_slices': num_slices,
        'num_pos': len(positive_slices),
        'num_neg': len(sampled_negatives),
        'recon_dice': recon_dice
    }

    if return_preview_data:
        preview_data = {
            'selected_slices': selected_slices,
            'orig_consensus': orig_consensus_mask,
            'resamp_consensus': crop_con,
            'spatial_meta': spatial_meta
        }
    else:
        preview_data = None

    del crop_vol, crop_con, crop_ses, orig_consensus_mask, orig_session_masks, res_con
    gc.collect()

    return manifest_entries, audit_record, summary_info, preview_data


def save_verification_preview_4col(selected_slices, orig_consensus, resamp_consensus, spatial_meta, clean_id, out_png):
    """
    Pipeline Step 9: Automated Visual Verification PNG (4-Column Image-Mask Audit)
    """
    pos_samples = [s for s in selected_slices if s['has_tumor'] == 1]
    neg_samples = [s for s in selected_slices if s['has_tumor'] == 0]

    show_pos = pos_samples[:2]
    show_neg = neg_samples[:2]
    show_all = show_pos + show_neg

    if not show_all:
        return

    n_rows = len(show_all)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    fig.suptitle(f"Medical Pipeline Verification Preview: {clean_id} (Canonical + Resampled 1.0mm³)", fontsize=14, fontweight='bold')

    recon_full = reverse_transform_mask(resamp_consensus, spatial_meta)

    for r_idx, r in enumerate(show_all):
        ax0 = axes[r_idx, 0] if n_rows > 1 else axes[0]
        ax1 = axes[r_idx, 1] if n_rows > 1 else axes[1]
        ax2 = axes[r_idx, 2] if n_rows > 1 else axes[2]
        ax3 = axes[r_idx, 3] if n_rows > 1 else axes[3]

        # Col 1: Cropped Resampled CT Image
        ax0.imshow(r['image_slice'], cmap='gray', vmin=0.0, vmax=1.0)
        label_t = "POSITIVE (Tumor)" if r['has_tumor'] == 1 else "NEGATIVE"
        ax0.set_title(f"Slice {r['slice_idx']} - {label_t}", fontsize=10)
        ax0.axis('off')

        # Col 2: Resampled Consensus Mask Overlay
        ax1.imshow(r['image_slice'], cmap='gray', vmin=0.0, vmax=1.0)
        if r['has_tumor'] == 1:
            ax1.imshow(np.ma.masked_where(r['mask_slice'] == 0, r['mask_slice']), cmap='spring', alpha=0.6)
            ax1.set_title(f"Resampled Mask ({r['tumor_pixels']} px)", fontsize=10, color='red')
        else:
            ax1.set_title("Resampled Mask: Empty", fontsize=10)
        ax1.axis('off')

        # Col 3: Radiologist Session Overlays
        ax2.imshow(r['image_slice'], cmap='gray', vmin=0.0, vmax=1.0)
        num_sess = r['session_slices'].shape[0]
        colors = ['cyan', 'magenta', 'yellow', 'lime']
        for s_i in range(num_sess):
            s_m = r['session_slices'][s_i]
            if np.sum(s_m) > 0:
                ax2.contour(s_m, levels=[0.5], colors=[colors[s_i % 4]], linewidths=1.2)
        ax2.set_title(f"{num_sess} Radiologists Contours", fontsize=10, color='blue')
        ax2.axis('off')

        # Col 4: Inverse Reconstruction Audit Validation
        # Map slice index back to original Z space
        orig_z = min(int(r['slice_idx'] / spatial_meta['zoom_factors'][2]), recon_full.shape[2] - 1)
        ax3.imshow(recon_full[:, :, orig_z], cmap='gray')
        ax3.set_title(f"Inverse DICOM Space (Z={orig_z})", fontsize=10, color='green')
        ax3.axis('off')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved medical verification preview to: {out_png}")


def process_patient_wrapper(args_tuple):
    """
    Wrapper function for ProcessPoolExecutor parallel execution.
    """
    pid, dataset_dir, output_dir, xml_index, split, crop_padding, lung_z_trim_ratio, keep_all_slices, save_preview_data = args_tuple
    should_keep_all = keep_all_slices or (split in ['val', 'test'])
    try:
        entries, audit_record, summary_info, preview_data = process_single_patient(
            pid,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            xml_index=xml_index,
            crop_padding=crop_padding,
            lung_z_trim_ratio=lung_z_trim_ratio,
            keep_all_slices=should_keep_all,
            return_preview_data=save_preview_data
        )
        for entry in entries:
            entry['split'] = split

        gc.collect()
        return True, pid, entries, audit_record, summary_info, preview_data, None
    except Exception as e:
        gc.collect()
        return False, pid, [], None, None, None, str(e)


def format_time(seconds):
    if seconds is None or math.isnan(seconds) or math.isinf(seconds):
        return "--:--"
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Preprocess LIDC-IDRI dataset into 1-channel 2D .npz slices with full 9-step medical pipeline.")
    parser.add_argument("--patient", "-p", type=str, default=None, help="Single Patient ID to process (e.g. LIDC-IDRI-0001)")
    parser.add_argument("--all", "-a", action="store_true", help="Process ALL patients in datasetmare")
    parser.add_argument("--max_patients", type=int, default=None, help="Limit total number of patients to process")

    parser.add_argument("--num_workers", "-w", type=int, default=DEFAULT_NUM_WORKERS, help=f"Number of parallel CPU worker processes (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR, help=f"Path to dataset directory (default: {DEFAULT_DATASET_DIR})")

    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Output directory for .npz files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--crop_padding", type=int, default=DEFAULT_CROP_PADDING, help=f"Margin padding in voxels for lung crop bounding box (default: {DEFAULT_CROP_PADDING})")
    parser.add_argument("--lung_z_trim_ratio", type=float, default=DEFAULT_LUNG_Z_TRIM_RATIO, help=f"Ratio of peak internal lung air threshold for Z-slice trimming (default: {DEFAULT_LUNG_Z_TRIM_RATIO})")
    parser.add_argument("--keep_all_slices", action="store_true", default=False, help="Keep all cropped Z-slices without negative sampling (always active for val/test splits)")
    parser.add_argument("--preview", action="store_true", default=False, help="Save visual verification preview image")

    args = parser.parse_args()

    xml_index = build_xml_index(args.dataset_dir)
    lidc_dir = os.path.join(args.dataset_dir, "lidc_idri")

    if args.patient:
        patient_dirs = [os.path.join(lidc_dir, args.patient)]
    else:
        patient_dirs = sorted(glob.glob(os.path.join(lidc_dir, "LIDC-IDRI-*")))

    if not patient_dirs:
        print(f"No patient directories found in {lidc_dir}")
        return

    if args.max_patients:
        patient_dirs = patient_dirs[:args.max_patients]

    num_workers = min(args.num_workers or 4, len(patient_dirs))
    print(f"\n=========================================================================")
    print(f" Starting LIDC Preprocessing: {len(patient_dirs)} Patients | {num_workers} Parallel CPU Workers")
    print(f" Output Directory: {args.output_dir}")
    print(f"=========================================================================\n")

    np.random.seed(42)
    shuffled_pids = [os.path.basename(p) for p in patient_dirs]
    np.random.shuffle(shuffled_pids)

    n_total = len(shuffled_pids)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    patient_splits = {}
    for i, pid in enumerate(shuffled_pids):
        if i < n_train:
            patient_splits[pid] = 'train'
        elif i < n_train + n_val:
            patient_splits[pid] = 'val'
        else:
            patient_splits[pid] = 'test'

    tasks = [
        (pid, args.dataset_dir, args.output_dir, xml_index, patient_splits[pid], args.crop_padding, args.lung_z_trim_ratio, args.keep_all_slices, args.preview)
        for pid in shuffled_pids
    ]

    all_manifest_entries = []
    all_audit_entries = []
    successful_patients = 0
    failed_patients = 0
    start_time = time.time()
    total_tasks = len(tasks)
    processed_count = 0

    if len(tasks) == 1:
        res = process_patient_wrapper(tasks[0])
        success, pid, entries, audit_record, summary_info, preview_data, err = res
        if success:
            all_manifest_entries.extend(entries)
            if audit_record:
                all_audit_entries.append(audit_record)
            successful_patients += 1
            pos = summary_info['num_pos']
            neg = summary_info['num_neg']
            dice = summary_info['recon_dice']
            split = patient_splits.get(pid, 'N/A')
            print(f"[ 1/1 | 100.0% | Elapsed: 00:00 ] {pid:<15s} [{split:<5s}] -> Saved {pos+neg:3d} slices ({pos:2d} Pos, {neg:3d} Neg) | Dice: {dice:.4f}")
            if args.preview and preview_data:
                preview_png = f"preprocess_preview_{pid}.png"
                save_verification_preview_4col(
                    preview_data['selected_slices'],
                    preview_data['orig_consensus'],
                    preview_data['resamp_consensus'],
                    preview_data['spatial_meta'],
                    pid,
                    preview_png
                )
        else:
            print(f"FAILED {pid}: {err}")
            failed_patients += 1
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_pid = {executor.submit(process_patient_wrapper, t): t[0] for t in tasks}

            for future in as_completed(future_to_pid):
                pid = future_to_pid.pop(future)
                processed_count += 1
                elapsed = time.time() - start_time
                pct = (processed_count / total_tasks) * 100
                avg_per_task = elapsed / processed_count
                eta = avg_per_task * (total_tasks - processed_count)

                time_str = f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}"
                split = patient_splits.get(pid, 'N/A')

                try:
                    success, pid, entries, audit_record, summary_info, preview_data, err = future.result()
                    if success:
                        all_manifest_entries.extend(entries)
                        if audit_record:
                            all_audit_entries.append(audit_record)
                        successful_patients += 1

                        pos = summary_info['num_pos']
                        neg = summary_info['num_neg']
                        dice = summary_info['recon_dice']
                        print(
                            f"[{processed_count:4d}/{total_tasks:4d} | {pct:5.1f}% | {time_str}] "
                            f"{pid:<15s} [{split:<5s}] -> Saved {pos+neg:3d} slices ({pos:2d} Pos, {neg:3d} Neg) | Recon Dice: {dice:.4f}"
                        )

                        if args.preview and preview_data and successful_patients == 1:
                            preview_png = f"preprocess_preview_{pid}.png"
                            save_verification_preview_4col(
                                preview_data['selected_slices'],
                                preview_data['orig_consensus'],
                                preview_data['resamp_consensus'],
                                preview_data['spatial_meta'],
                                pid,
                                preview_png
                            )
                    else:
                        print(f"[{processed_count:4d}/{total_tasks:4d} | {pct:5.1f}% | {time_str}] {pid:<15s} [{split:<5s}] -> FAILED: {err}")
                        failed_patients += 1
                except Exception as exc:
                    print(f"[{processed_count:4d}/{total_tasks:4d} | {pct:5.1f}% | {time_str}] {pid:<15s} [{split:<5s}] -> EXCEPTION: {exc}")
                    failed_patients += 1

                if processed_count % 50 == 0:
                    gc.collect()

    if all_audit_entries:
        df_audit = pd.DataFrame(all_audit_entries)
        audit_csv = os.path.join(args.output_dir, "patient_series_audit.csv").replace("\\", "/")
        os.makedirs(args.output_dir, exist_ok=True)
        df_audit.to_csv(audit_csv, index=False)

    if all_manifest_entries:
        df_manifest = pd.DataFrame(all_manifest_entries)
        manifest_csv = os.path.join(args.output_dir, "dataset_manifest.csv").replace("\\", "/")
        os.makedirs(args.output_dir, exist_ok=True)
        df_manifest.to_csv(manifest_csv, index=False)

        num_pos_total = len(df_manifest[df_manifest['has_tumor'] == 1])
        num_neg_total = len(df_manifest[df_manifest['has_tumor'] == 0])

        print(f"\n=======================================================")
        print(f"Medical Preprocessing Pipeline Complete!")
        print(f"Successfully processed {successful_patients}/{len(tasks)} patients ({failed_patients} failed)")
        print(f"Total .npz slices saved: {len(df_manifest)} ({num_pos_total} Positive, {num_neg_total} Negative)")
        print(f"Saved master dataset manifest to: {manifest_csv}")
        if all_audit_entries:
            print(f"Saved patient & series audit report to: {audit_csv}")
        print(f"=======================================================\n")

if __name__ == "__main__":
    main()
