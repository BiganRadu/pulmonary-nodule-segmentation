import os
import sys
import glob
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.path import Path
from scipy.ndimage import binary_fill_holes

def build_xml_index(datasetmare_dir=r"E:\CS2023-2027\AIMAS\practica\datasetmare"):
    """
    Indexes XML files in datasetmare by StudyInstanceUID, SeriesInstanceUID, and imageSOP_UIDs.
    """
    xml_dir = os.path.join(datasetmare_dir, "tcia-lidc-xml")
    if not os.path.exists(xml_dir):
        xml_dir = datasetmare_dir

    xml_files = glob.glob(os.path.join(xml_dir, "**/*.xml"), recursive=True)
    single_xml = os.path.join(datasetmare_dir, "161-resubmitted-correction-3-9-12.xml")
    if os.path.exists(single_xml) and single_xml not in xml_files:
        xml_files.append(single_xml)

    index = {'study': {}, 'series': {}, 'sop': {}}
    ns = {'nih': 'http://www.nih.gov'}

    print(f"Indexing {len(xml_files)} XML annotation files from {xml_dir}...")
    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            header = root.find('nih:ResponseHeader', ns)
            if header is None:
                header = root

            study_uid = header.findtext('nih:StudyInstanceUID', default='', namespaces=ns).strip()
            series_uid = header.findtext('nih:SeriesInstanceUID', default='', namespaces=ns).strip()

            if study_uid:
                index['study'][study_uid] = xf
            if series_uid:
                index['series'][series_uid] = xf

            for sop_elem in root.findall('.//nih:imageSOP_UID', ns):
                if sop_elem.text:
                    sop_clean = sop_elem.text.strip()
                    index['sop'][sop_clean] = xf
        except Exception:
            continue

    print(f"XML Index ready: {len(index['study'])} studies, {len(index['sop'])} SOPs.")
    return index

def load_patient_dicom_and_xml(patient_id, datasetmare_dir=r"E:\CS2023-2027\AIMAS\practica\datasetmare", xml_index=None):
    """
    Loads raw DICOM volume for a patient and matches its XML expert annotation file.
    """
    lidc_dir = os.path.join(datasetmare_dir, "lidc_idri")
    if not os.path.exists(lidc_dir):
        lidc_dir = datasetmare_dir




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

    # Group DICOM files by SeriesInstanceUID & select main 3D CT scan
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

            series_groups[series_uid].append({
                'path': f,
                'z': z_pos,
                'sop': sop_uid,
                'study_uid': study_uid,
                'series_uid': series_uid,
                'slope': slope,
                'intercept': intercept,
                'rows': rows,
                'cols': cols
            })
        except Exception:
            continue

    if not series_groups:
        raise ValueError(f"No valid CT series found in {patient_path}")

    primary_series_uid = max(series_groups.keys(), key=lambda s_uid: len(series_groups[s_uid]))
    slices_info = series_groups[primary_series_uid]
    slices_info.sort(key=lambda s: s['z'])

    rows, cols = slices_info[0]['rows'], slices_info[0]['cols']
    num_z = len(slices_info)
    volume = np.zeros((rows, cols, num_z), dtype=np.float32)

    for z_idx, s in enumerate(slices_info):
        ds = pydicom.dcmread(s['path'])
        pixel_array = ds.pixel_array.astype(np.float32)
        hu_pixels = pixel_array * s['slope'] + s['intercept']
        volume[:, :, z_idx] = hu_pixels

    # Match XML annotation file
    matched_xml = None
    if xml_index:
        study_uid = slices_info[0]['study_uid']
        series_uid = slices_info[0]['series_uid']

        if series_uid in xml_index['series']:
            matched_xml = xml_index['series'][series_uid]
        elif study_uid in xml_index['study']:
            matched_xml = xml_index['study'][study_uid]
        else:
            for s in slices_info:
                if s['sop'] in xml_index['sop']:
                    matched_xml = xml_index['sop'][s['sop']]
                    break

    return volume, slices_info, matched_xml, clean_id

def parse_consensus_mask(xml_file, slices_info):
    """
    Parses LIDC XML expert annotations and constructs a 50% majority consensus 3D binary mask.
    Uses OpenCV cv2.fillPoly and direct uint8 accumulation for memory efficiency.
    """
    rows, cols = slices_info[0]['rows'], slices_info[0]['cols']
    num_z = len(slices_info)
    consensus_mask = np.zeros((rows, cols, num_z), dtype=np.uint8)

    if not xml_file or not os.path.exists(xml_file):
        return consensus_mask

    sop_to_zidx = {s['sop']: i for i, s in enumerate(slices_info)}

    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'nih': 'http://www.nih.gov'}

    sessions = root.findall('nih:readingSession', ns)
    if not sessions:
        return consensus_mask

    consensus_count = np.zeros((rows, cols, num_z), dtype=np.uint8)

    for s in sessions:
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

        consensus_count += (sess_mask > 0).astype(np.uint8)

    min_agreement = max(1, len(sessions) // 2)
    consensus_mask = (consensus_count >= min_agreement).astype(np.uint8)
    return consensus_mask

def normalize_lung_window(hu_volume, min_hu=-1000.0, max_hu=400.0):
    """
    Clips CT volume to lung window [-1000, 400] HU and normalizes to [0.0, 1.0].
    """
    clipped = np.clip(hu_volume, min_hu, max_hu)
    normalized = (clipped - min_hu) / (max_hu - min_hu)
    return normalized.astype(np.float32)

def process_single_patient(patient_id, datasetmare_dir=r"E:\CS2023-2027\AIMAS\practica\datasetmare", output_dir="preprocessed_data", xml_index=None, min_air_ratio=0.05, neg_pos_ratio=1.5, max_neg_slices=20):

    """
    Preprocesses a single patient:
    - Normalizes CT HU volume
    - Builds consensus target masks
    - Filters out non-lung slices (neck/abdomen)
    - Mines all positive slices and a balanced subset of negative lung slices
    - Saves 1-channel .npz files and returns manifest records
    """
    volume, slices_info, matched_xml, clean_id = load_patient_dicom_and_xml(patient_id, datasetmare_dir, xml_index)
    consensus_mask = parse_consensus_mask(matched_xml, slices_info)
    norm_volume = normalize_lung_window(volume)

    num_slices = norm_volume.shape[2]
    positive_slices = []
    negative_slices = []

    for z in range(num_slices):
        slice_img = norm_volume[:, :, z]  # (512, 512)
        slice_mask = consensus_mask[:, :, z]  # (512, 512)

        # Calculate internal lung air pixels (inside body mask) to ignore external air & abdominal bowel gas
        hu_slice = volume[:, :, z]
        body_mask = binary_fill_holes(hu_slice > -500.0)
        internal_air_pixels = np.sum(((hu_slice >= -1000.0) & (hu_slice <= -400.0)) & body_mask)
        body_area = max(1, np.sum(body_mask))
        internal_air_ratio = internal_air_pixels / body_area
        tumor_pixels = int(np.sum(slice_mask > 0))

        # Always keep positive tumor slices.
        # For negative background slices, filter aggressively: require >= 20,000 internal air pixels (core lung cavity)
        if tumor_pixels == 0 and internal_air_pixels < 20000:
            continue

        record = {
            'patient_id': clean_id,
            'slice_idx': z,
            'has_tumor': 1 if tumor_pixels > 0 else 0,
            'tumor_pixels': tumor_pixels,
            'air_ratio': round(float(internal_air_ratio), 4),
            'image_slice': slice_img,
            'mask_slice': slice_mask
        }

        if tumor_pixels > 0:
            positive_slices.append(record)
        else:
            negative_slices.append(record)

    # Sample balanced negative slices
    num_pos = len(positive_slices)
    if num_pos > 0:
        num_neg_to_keep = min(len(negative_slices), int(num_pos * neg_pos_ratio) + 2)
    else:
        num_neg_to_keep = min(len(negative_slices), max_neg_slices)

    if len(negative_slices) > num_neg_to_keep:
        # Uniformly sample negative slices across the lung volume
        indices = np.linspace(0, len(negative_slices) - 1, num_neg_to_keep, dtype=int)
        sampled_negatives = [negative_slices[i] for i in indices]
    else:
        sampled_negatives = negative_slices

    selected_slices = positive_slices + sampled_negatives
    selected_slices.sort(key=lambda r: r['slice_idx'])

    # Save to .npz files
    patient_out_dir = os.path.join(output_dir, clean_id)
    os.makedirs(patient_out_dir, exist_ok=True)

    manifest_entries = []
    for r in selected_slices:
        filename = f"{clean_id}_slice{r['slice_idx']:03d}.npz"
        filepath = os.path.join(patient_out_dir, filename).replace("\\", "/")

        # Reshape to 1-channel 2D tensors: (1, 512, 512)
        img_tensor = np.expand_dims(r['image_slice'], axis=0).astype(np.float32)
        mask_tensor = np.expand_dims(r['mask_slice'], axis=0).astype(np.uint8)

        np.savez_compressed(filepath, image=img_tensor, mask=mask_tensor)

        manifest_entries.append({
            'patient_id': clean_id,
            'slice_idx': r['slice_idx'],
            'has_tumor': r['has_tumor'],
            'tumor_pixels': r['tumor_pixels'],
            'air_ratio': r['air_ratio'],
            'filepath': filepath
        })

    print(f"[{clean_id}] Total slices: {num_slices} | Valid lung slices: {len(positive_slices) + len(negative_slices)} | Saved: {len(selected_slices)} ({len(positive_slices)} Positive, {len(sampled_negatives)} Negative)")
    return manifest_entries, selected_slices

def save_verification_preview(selected_slices, clean_id, out_png):
    """
    Saves a visual verification preview PNG showing extracted positive & negative slices.
    """
    pos_samples = [s for s in selected_slices if s['has_tumor'] == 1]
    neg_samples = [s for s in selected_slices if s['has_tumor'] == 0]

    # Select up to 3 positive and 3 negative slices
    show_pos = pos_samples[:3]
    show_neg = neg_samples[:3]
    show_all = show_pos + show_neg

    if not show_all:
        print("No slices to plot in preview.")
        return

    n_cols = len(show_all)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))
    fig.suptitle(f"Preprocessing Verification Preview: {clean_id} (1-Channel 2D .npz)", fontsize=14, fontweight='bold')

    for c, r in enumerate(show_all):
        img_ax = axes[0, c] if n_cols > 1 else axes[0]
        mask_ax = axes[1, c] if n_cols > 1 else axes[1]

        # Normalized CT Image
        img_ax.imshow(r['image_slice'], cmap='gray', vmin=0.0, vmax=1.0)
        label_type = "POSITIVE (Tumor)" if r['has_tumor'] == 1 else "NEGATIVE (Background)"
        img_ax.set_title(f"Slice {r['slice_idx']}\n{label_type}", fontsize=10, color='red' if r['has_tumor'] == 1 else 'black')
        img_ax.axis('off')

        # Mask overlay
        mask_ax.imshow(r['image_slice'], cmap='gray', vmin=0.0, vmax=1.0)
        if r['has_tumor'] == 1:
            mask_ax.imshow(np.ma.masked_where(r['mask_slice'] == 0, r['mask_slice']), cmap='spring', alpha=0.6)
            mask_ax.set_title(f"Tumor Voxels: {r['tumor_pixels']}", fontsize=10, color='red')
        else:
            mask_ax.set_title("Mask: Empty (0)", fontsize=10)
        mask_ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved verification preview PNG to: {out_png}")

def _patient_worker(task):
    """
    Worker function for parallel multiprocessing execution.
    """
    pid, datasetmare_dir, output_dir, split, xml_index = task
    try:
        entries, selected_slices = process_single_patient(
            patient_id=pid,
            datasetmare_dir=datasetmare_dir,
            output_dir=os.path.join(output_dir, split),
            xml_index=xml_index
        )
        for entry in entries:
            entry['split'] = split
        return True, pid, entries, selected_slices, None
    except Exception as e:
        return False, pid, [], [], str(e)

def main():
    parser = argparse.ArgumentParser(description="Preprocess LIDC-IDRI dataset into 1-channel 2D .npz slices and target masks.")
    parser.add_argument("--patient", "-p", type=str, default=None, help="Single Patient ID to process (e.g. LIDC-IDRI-0001)")
    parser.add_argument("--all", "-a", action="store_true", help="Process ALL patients in datasetmare")
    parser.add_argument("--max_patients", type=int, default=None, help="Limit total number of patients to process")
    parser.add_argument("--num_workers", "-w", type=int, default=8, help="Number of parallel CPU worker processes (defaults to safe RAM limit of 6)")
    parser.add_argument("--datasetmare_dir", type=str, default=r"E:\CS2023-2027\AIMAS\practica\datasetmare", help="Path to dataset directory (default: E:\\CS2023-2027\\AIMAS\\practica\\datasetmare)")
    parser.add_argument("--output_dir", type=str, default="preprocessed_data", help="Output directory for .npz files")
    parser.add_argument("--preview", action="store_true", default=False, help="Save visual verification preview image")

    args = parser.parse_args()

    xml_index = build_xml_index(args.datasetmare_dir)

    lidc_dir = os.path.join(args.datasetmare_dir, "lidc_idri")
    if not os.path.exists(lidc_dir):
        lidc_dir = args.datasetmare_dir

    if args.patient:
        patient_dirs = [p for p in [os.path.join(lidc_dir, args.patient)] if os.path.isdir(p)]
    else:
        patient_dirs = [p for p in sorted(glob.glob(os.path.join(lidc_dir, "LIDC-IDRI-*"))) if os.path.isdir(p)]

    if not patient_dirs:
        print(f"No patient directories found in {lidc_dir}")
        return


    if args.max_patients:
        patient_dirs = patient_dirs[:args.max_patients]

    num_workers = min(args.num_workers or 4, len(patient_dirs))
    print(f"\n--- Found {len(patient_dirs)} patient directories to process using {num_workers} parallel CPU workers ---")

    # Patient-level Train (80%) / Val (10%) / Test (10%) split assignment
    np.random.seed(42)
    patient_ids = [os.path.basename(p) for p in patient_dirs]
    shuffled_ids = patient_ids.copy()
    np.random.shuffle(shuffled_ids)

    n_patients = len(shuffled_ids)
    n_train = int(0.80 * n_patients)
    n_val = int(0.10 * n_patients)

    split_map = {}
    for idx, pid in enumerate(shuffled_ids):
        if idx < n_train:
            split_map[pid] = "train"
        elif idx < n_train + n_val:
            split_map[pid] = "val"
        else:
            split_map[pid] = "test"

    tasks = [
        (os.path.basename(ppath), args.datasetmare_dir, args.output_dir, split_map.get(os.path.basename(ppath), "train"), xml_index)
        for ppath in patient_dirs
    ]

    all_manifest_entries = []
    success_count = 0
    fail_count = 0

    if num_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_pid = {executor.submit(_patient_worker, t): t[0] for t in tasks}
            done_count = 0
            for future in as_completed(future_to_pid):
                done_count += 1
                success, pid, entries, selected_slices, err = future.result()
                if success:
                    success_count += 1
                    all_manifest_entries.extend(entries)
                    if args.preview and done_count == 1:
                        preview_png = os.path.join(args.output_dir, f"{pid}_verification_preview.png")
                        save_verification_preview(selected_slices, pid, preview_png)
                    print(f"[{done_count}/{len(tasks)}] Finished {pid} -> Saved {len(entries)} slices")
                else:
                    fail_count += 1
                    print(f"[{done_count}/{len(tasks)}] ERROR processing {pid}: {err}")
    else:
        for idx, t in enumerate(tasks, 1):
            success, pid, entries, selected_slices, err = _patient_worker(t)
            if success:
                success_count += 1
                all_manifest_entries.extend(entries)
                if args.preview and idx == 1:
                    preview_png = os.path.join(args.output_dir, f"{pid}_verification_preview.png")
                    save_verification_preview(selected_slices, pid, preview_png)
                print(f"[{idx}/{len(tasks)}] Finished {pid} -> Saved {len(entries)} slices")
            else:
                fail_count += 1
                print(f"[{idx}/{len(tasks)}] ERROR processing {pid}: {err}")

    # Save Master Manifest CSV
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_df = pd.DataFrame(all_manifest_entries)
    master_manifest_path = os.path.join(args.output_dir, "dataset_manifest.csv")
    manifest_df.to_csv(master_manifest_path, index=False)

    print("\n=======================================================")
    print(f"Preprocessing Complete!")
    print(f"Successfully processed: {success_count} patients")
    if fail_count > 0:
        print(f"Failed: {fail_count} patients")
    print(f"Total Slices Saved: {len(manifest_df)}")
    if not manifest_df.empty:
        pos_count = (manifest_df['has_tumor'] == 1).sum()
        neg_count = (manifest_df['has_tumor'] == 0).sum()
        print(f"  - Positive Tumor Slices: {pos_count} ({pos_count / len(manifest_df) * 100:.1f}%)")
        print(f"  - Negative Lung Slices:  {neg_count} ({neg_count / len(manifest_df) * 100:.1f}%)")
        print(f"Master Manifest CSV saved to: {master_manifest_path}")
    print("=======================================================")

if __name__ == "__main__":
    main()
