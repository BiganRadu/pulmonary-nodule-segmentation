# Standard library
import os
import sys
import glob
import argparse
import xml.etree.ElementTree as ET

# Windows-specific environment configuration
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 3rd party
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

# Default Configuration Constants
DEFAULT_DATASET_DIR = "/mnt/hdd/CS2023-2027/AIMAS/practica/datasetmare"
DEFAULT_PATIENT_ID = "LIDC-IDRI-0001"

def build_xml_index(datasetmare_dir):
    """
    Indexes XML files in datasetmare by StudyInstanceUID, SeriesInstanceUID, and imageSOP_UIDs for fast lookup.
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

    print(f"Indexing {len(xml_files)} XML annotation files...")
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

            # Index SOPs inside ROIs
            for sop_elem in root.findall('.//nih:imageSOP_UID', ns):
                if sop_elem.text:
                    sop_clean = sop_elem.text.strip()
                    index['sop'][sop_clean] = xf
        except Exception:
            continue

    print(f"XML Index ready: {len(index['study'])} studies, {len(index['series'])} series, {len(index['sop'])} SOPs.")
    return index

def load_patient_dicom_and_xml(patient_arg, datasetmare_dir="datasetmare", xml_index=None):
    """
    Loads raw DICOM volume for a patient and matches its XML expert annotation file.
    Groups DICOM slices by SeriesInstanceUID to pick the main CT scan volume.
    """
    lidc_dir = os.path.join(datasetmare_dir, "lidc_idri")

    # Clean patient ID
    patient_id = os.path.basename(patient_arg.strip('/\\'))
    if not patient_id.startswith("LIDC-IDRI-"):
        if patient_id.isdigit():
            patient_id = f"LIDC-IDRI-{int(patient_id):04d}"
        else:
            patient_id = f"LIDC-IDRI-{patient_id}"

    patient_path = os.path.join(lidc_dir, patient_id)
    if not os.path.exists(patient_path):
        matches = glob.glob(os.path.join(lidc_dir, f"*{patient_id}*"))
        if matches:
            patient_path = matches[0]
            patient_id = os.path.basename(patient_path)
        else:
            raise FileNotFoundError(f"Could not find patient directory for '{patient_arg}' in '{lidc_dir}'")

    print(f"Loading DICOM files for {patient_id} from {patient_path}...")
    dcm_files = glob.glob(os.path.join(patient_path, "**/*.dcm"), recursive=True)
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {patient_path}")

    # Group DICOM files by SeriesInstanceUID
    series_groups = {}
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            series_uid = str(getattr(ds, 'SeriesInstanceUID', 'default_series'))
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
        raise ValueError(f"Failed to parse DICOM headers in {patient_path}")

    # Select primary 3D CT series (the series with the maximum number of slices, typically 100-400 slices)
    primary_series_uid = max(series_groups.keys(), key=lambda s_uid: len(series_groups[s_uid]))
    slices_info = series_groups[primary_series_uid]

    # Sort slices by Z position ascending
    slices_info.sort(key=lambda s: s['z'])

    print(f"Selected Series UID: {primary_series_uid[:30]}... ({len(slices_info)} 3D slices)")

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

    return volume, slices_info, matched_xml, patient_id

def parse_lidc_xml(xml_file, slices_info):
    """
    Parses LIDC XML expert annotations.
    Returns:
        masks_by_session: dict mapping radiologist session index (0-3) to 3D binary mask array (512, 512, Z)
        consensus_mask: 3D binary mask where >= 50% of radiologists annotated a nodule
        nodule_attributes: list of dicts summarizing expert ratings (malignancy, sphericity, etc.)
    """
    if not xml_file or not os.path.exists(xml_file):
        return {}, None, []

    sop_to_zidx = {s['sop']: i for i, s in enumerate(slices_info)}
    rows, cols = slices_info[0]['rows'], slices_info[0]['cols']
    num_z = len(slices_info)

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    grid_points = np.vstack((x.flatten(), y.flatten())).T

    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'nih': 'http://www.nih.gov'}

    sessions = root.findall('nih:readingSession', ns)
    masks_by_session = {}
    nodule_attributes = []

    for sess_idx, s in enumerate(sessions):
        sess_mask = np.zeros((rows, cols, num_z), dtype=np.uint8)
        nodules = s.findall('nih:unblindedReadNodule', ns)

        for nod_idx, nod in enumerate(nodules):
            # Parse characteristics
            char_elem = nod.find('nih:characteristics', ns)
            attr = {'session': sess_idx, 'nodule': nod_idx}
            if char_elem is not None:
                for child in char_elem:
                    tag_name = child.tag.split('}')[-1]
                    attr[tag_name] = int(child.text) if child.text and child.text.isdigit() else child.text
                nodule_attributes.append(attr)

            # Parse ROIs / polygons
            for roi in nod.findall('nih:roi', ns):
                sop = roi.findtext('nih:imageSOP_UID', default='', namespaces=ns).strip()
                if sop in sop_to_zidx:
                    z_idx = sop_to_zidx[sop]
                    edges = roi.findall('nih:edgeMap', ns)
                    if len(edges) >= 3:
                        poly = [(int(e.findtext('nih:xCoord', namespaces=ns)), int(e.findtext('nih:yCoord', namespaces=ns))) for e in edges]
                        poly_path = Path(poly)
                        in_poly = poly_path.contains_points(grid_points).reshape((rows, cols))
                        sess_mask[:, :, z_idx] |= in_poly.astype(np.uint8)

        masks_by_session[sess_idx] = sess_mask

    num_sessions = len(masks_by_session)
    if num_sessions > 0:
        session_stack = np.stack(list(masks_by_session.values()), axis=0)
        consensus_count = np.sum(session_stack, axis=0)
        min_agreement = max(1, num_sessions // 2)
        consensus_mask = (consensus_count >= min_agreement).astype(np.uint8)
    else:
        consensus_mask = np.zeros((rows, cols, num_z), dtype=np.uint8)

    return masks_by_session, consensus_mask, nodule_attributes

def get_nodule_center_and_bounds(mask_3d):
    """Returns center and bounding box for a 3D mask."""
    if mask_3d is None or np.sum(mask_3d > 0) == 0:
        return None

    nonzero = np.argwhere(mask_3d > 0)
    min_c = nonzero.min(axis=0)  # y, x, z
    max_c = nonzero.max(axis=0)
    center_c = (min_c + max_c) // 2

    return {
        'min': min_c,
        'max': max_c,
        'center': center_c,
        'count': len(nonzero)
    }

def save_lidc_grid_preview(volume, consensus_mask, masks_by_session, output_path, patient_id="LIDC-IDRI"):
    """
    Saves a 3x3 grid preview of the CT volume and expert radiologist nodule annotations.
    """
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    fig.suptitle(f"LIDC-IDRI Expert Annotations Preview: {patient_id}", fontsize=16, fontweight='bold')

    shape = volume.shape
    mask_info = get_nodule_center_and_bounds(consensus_mask)

    if mask_info:
        cy, cx, cz = mask_info['center']
        min_y, max_y = mask_info['min'][0], mask_info['max'][0]
        min_x, max_x = mask_info['min'][1], mask_info['max'][1]
        min_z, max_z = mask_info['min'][2], mask_info['max'][2]

        z_slices = [min_z, cz, max_z]
        y_slices = [min_y, cy, max_y]
        x_slices = [min_x, cx, max_x]
    else:
        z_slices = [int(shape[2] * p) for p in (0.25, 0.50, 0.75)]
        y_slices = [int(shape[0] * p) for p in (0.25, 0.50, 0.75)]
        x_slices = [int(shape[1] * p) for p in (0.25, 0.50, 0.75)]

    # CT Lung windowing: Center = -600, Width = 1500
    win_min, win_max = -1350, 150
    session_colors = ['red', 'cyan', 'yellow', 'lime']

    # Row 0: Axial (XY plane, z slice)
    for col, z in enumerate(z_slices):
        ax = axes[0, col]
        slice_img = volume[:, :, z]
        ax.imshow(slice_img, cmap='gray', vmin=win_min, vmax=win_max)

        for s_idx, s_mask in masks_by_session.items():
            if np.sum(s_mask[:, :, z]) > 0:
                color = session_colors[s_idx % len(session_colors)]
                ax.contour(s_mask[:, :, z], levels=[0.5], colors=[color], linewidths=1.5)

        if consensus_mask is not None and np.sum(consensus_mask[:, :, z]) > 0:
            ax.imshow(np.ma.masked_where(consensus_mask[:, :, z] == 0, consensus_mask[:, :, z]), cmap='spring', alpha=0.5)

        ax.set_title(f"Axial (Z={z}/{shape[2]})")
        ax.axis('off')

    # Row 1: Coronal (XZ plane, y slice)
    for col, y in enumerate(y_slices):
        ax = axes[1, col]
        slice_img = volume[y, :, :]
        ax.imshow(np.rot90(slice_img), cmap='gray', vmin=win_min, vmax=win_max)

        if consensus_mask is not None and np.sum(consensus_mask[y, :, :]) > 0:
            c_slice = np.rot90(consensus_mask[y, :, :])
            ax.imshow(np.ma.masked_where(c_slice == 0, c_slice), cmap='spring', alpha=0.5)

        ax.set_title(f"Coronal (Y={y}/{shape[0]})")
        ax.axis('off')

    # Row 2: Sagittal (YZ plane, x slice)
    for col, x in enumerate(x_slices):
        ax = axes[2, col]
        slice_img = volume[:, x, :]
        ax.imshow(np.rot90(slice_img), cmap='gray', vmin=win_min, vmax=win_max)

        if consensus_mask is not None and np.sum(consensus_mask[:, x, :]) > 0:
            c_slice = np.rot90(consensus_mask[:, x, :])
            ax.imshow(np.ma.masked_where(c_slice == 0, c_slice), cmap='spring', alpha=0.5)

        ax.set_title(f"Sagittal (X={x}/{shape[1]})")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved LIDC preview grid image to: {output_path}")

class InteractiveLIDCViewer:
    """Interactive GUI viewer for LIDC-IDRI DICOM volumes with expert radiologist annotations."""
    def __init__(self, volume, consensus_mask, masks_by_session, attributes, patient_id=""):
        self.volume = volume
        self.consensus_mask = consensus_mask
        self.masks_by_session = masks_by_session
        self.attributes = attributes
        self.patient_id = patient_id
        self.shape = volume.shape

        self.mask_info = get_nodule_center_and_bounds(consensus_mask)
        self.view_axis = 2  # 2: Axial (Z), 1: Coronal (Y), 0: Sagittal (X)

        if self.mask_info is not None:
            self.slice_idx = self.mask_info['center'][2]
        else:
            self.slice_idx = self.shape[2] // 2

        self.show_consensus = True

        self.presets = {
            'Lung': (-600, 1500),
            'Soft Tissue': (40, 400),
            'Bone': (400, 1800),
            'Full Range': (float(volume.mean()), float(volume.max() - volume.min()))
        }
        self.current_preset = 'Lung'
        self.setup_gui()

    def get_current_slice(self):
        if self.view_axis == 2:
            slice_img = self.volume[:, :, self.slice_idx]
            c_mask = self.consensus_mask[:, :, self.slice_idx] if self.consensus_mask is not None else None
            ind_masks = {k: v[:, :, self.slice_idx] for k, v in self.masks_by_session.items()}
        elif self.view_axis == 1:
            slice_img = np.rot90(self.volume[self.slice_idx, :, :])
            c_mask = np.rot90(self.consensus_mask[self.slice_idx, :, :]) if self.consensus_mask is not None else None
            ind_masks = {k: np.rot90(v[self.slice_idx, :, :]) for k, v in self.masks_by_session.items()}
        else:
            slice_img = np.rot90(self.volume[:, self.slice_idx, :])
            c_mask = np.rot90(self.consensus_mask[:, self.slice_idx, :]) if self.consensus_mask is not None else None
            ind_masks = {k: np.rot90(v[:, self.slice_idx, :]) for k, v in self.masks_by_session.items()}

        return slice_img, c_mask, ind_masks

    def setup_gui(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        plt.subplots_adjust(left=0.1, bottom=0.25, top=0.9)

        slice_img, c_mask, ind_masks = self.get_current_slice()

        wc, ww = self.presets[self.current_preset]
        vmin, vmax = wc - ww/2, wc + ww/2
        self.im = self.ax.imshow(slice_img, cmap='gray', vmin=vmin, vmax=vmax)

        if c_mask is not None:
            self.mask_im = self.ax.imshow(
                np.ma.masked_where(c_mask == 0, c_mask),
                cmap='spring', alpha=0.5,
                visible=self.show_consensus
            )
        else:
            self.mask_im = None

        axis_names = {2: 'Axial (Z)', 1: 'Coronal (Y)', 0: 'Sagittal (X)'}
        title_extra = ""
        if c_mask is not None and np.sum(c_mask > 0) > 0:
            title_extra = f" | Expert Annotated Nodule Voxels: {int(np.sum(c_mask > 0))}"

        self.ax.set_title(f"Patient {self.patient_id} - {axis_names[self.view_axis]} Slice {self.slice_idx}/{self.shape[self.view_axis]}{title_extra}", fontsize=11)
        self.ax.axis('off')

        # Slider for slice position
        ax_slider = plt.axes([0.2, 0.12, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Slice', 0, self.shape[self.view_axis] - 1, valinit=self.slice_idx, valfmt='%d')
        self.slider.on_changed(self.update_slice)

        # Radio buttons for anatomical view plane
        ax_plane = plt.axes([0.05, 0.02, 0.22, 0.08])
        self.radio_plane = RadioButtons(ax_plane, ('Axial (Z)', 'Coronal (Y)', 'Sagittal (X)'), active=0)
        self.radio_plane.on_clicked(self.change_plane)

        # Radio buttons for HU window presets
        ax_window = plt.axes([0.30, 0.02, 0.22, 0.08])
        self.radio_window = RadioButtons(ax_window, ('Lung', 'Soft Tissue', 'Bone', 'Full Range'), active=0)
        self.radio_window.on_clicked(self.change_window)

        # Jump to Lesion Center Button
        if self.mask_info is not None:
            ax_jump_btn = plt.axes([0.55, 0.04, 0.20, 0.05])
            self.btn_jump = Button(ax_jump_btn, 'Jump to Nodule')
            self.btn_jump.on_clicked(self.jump_to_nodule)

        # Toggle CheckButtons
        if self.consensus_mask is not None:
            ax_toggle = plt.axes([0.78, 0.02, 0.18, 0.08])
            self.check_toggle = CheckButtons(ax_toggle, ['Consensus Mask'], [self.show_consensus])
            self.check_toggle.on_clicked(self.toggle_display)

        # Scroll wheel event listener
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def update_slice(self, val):
        self.slice_idx = int(self.slider.val)
        slice_img, c_mask, ind_masks = self.get_current_slice()

        self.im.set_data(slice_img)
        if self.mask_im is not None and c_mask is not None:
            self.mask_im.set_data(np.ma.masked_where(c_mask == 0, c_mask))

        axis_names = {2: 'Axial (Z)', 1: 'Coronal (Y)', 0: 'Sagittal (X)'}
        title_extra = ""
        if c_mask is not None and np.sum(c_mask > 0) > 0:
            title_extra = f" | Expert Annotated Nodule Voxels: {int(np.sum(c_mask > 0))}"

        self.ax.set_title(f"Patient {self.patient_id} - {axis_names[self.view_axis]} Slice {self.slice_idx}/{self.shape[self.view_axis]}{title_extra}", fontsize=11)
        self.fig.canvas.draw_idle()

    def change_plane(self, label):
        plane_map = {'Axial (Z)': 2, 'Coronal (Y)': 1, 'Sagittal (X)': 0}
        self.view_axis = plane_map[label]
        max_slice = self.shape[self.view_axis] - 1

        if self.mask_info is not None:
            axis_to_center = {2: self.mask_info['center'][2], 1: self.mask_info['center'][0], 0: self.mask_info['center'][1]}
            self.slice_idx = axis_to_center[self.view_axis]
        else:
            self.slice_idx = max_slice // 2

        self.slider.valmax = max_slice
        self.slider.ax.set_xlim(0, max_slice)
        self.slider.set_val(self.slice_idx)
        self.update_slice(self.slice_idx)

    def change_window(self, label):
        self.current_preset = label
        wc, ww = self.presets[label]
        self.im.set_clim(wc - ww/2, wc + ww/2)
        self.fig.canvas.draw_idle()

    def jump_to_nodule(self, event):
        if self.mask_info is None:
            return
        axis_to_center = {2: self.mask_info['center'][2], 1: self.mask_info['center'][0], 0: self.mask_info['center'][1]}
        self.slider.set_val(axis_to_center[self.view_axis])

    def toggle_display(self, label):
        if label == 'Consensus Mask' and self.mask_im is not None:
            self.show_consensus = not self.show_consensus
            self.mask_im.set_visible(self.show_consensus)
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.button == 'up':
            new_val = min(self.slice_idx + 1, self.shape[self.view_axis] - 1)
        elif event.button == 'down':
            new_val = max(self.slice_idx - 1, 0)
        else:
            return
        self.slider.set_val(new_val)

    def show(self):
        plt.show()

def list_lidc_patients(datasetmare_dir="datasetmare"):
    """Lists available LIDC-IDRI patient directories."""
    lidc_dir = os.path.join(datasetmare_dir, "lidc_idri")
    if not os.path.exists(lidc_dir):
        print(f"Directory {lidc_dir} not found.")
        return

    patients = sorted(os.listdir(lidc_dir))
    print(f"\nFound {len(patients)} LIDC-IDRI patients in '{lidc_dir}':")
    for p in patients[:25]:
        dcm_count = len(glob.glob(os.path.join(lidc_dir, p, "**/*.dcm"), recursive=True))
        print(f"  - {p} ({dcm_count} DICOM slices)")
    if len(patients) > 25:
        print(f"  ... and {len(patients) - 25} more patients.")

def main():
    parser = argparse.ArgumentParser(description="Visualize LIDC-IDRI DICOM CT Scans and Expert Radiologist XML Annotations")
    parser.add_argument("--patient", "-p", type=str, default=DEFAULT_PATIENT_ID, help=f"Patient ID (default: {DEFAULT_PATIENT_ID})")
    parser.add_argument("--dataset_dir", "-d", type=str, default=DEFAULT_DATASET_DIR, help=f"Path to dataset directory (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--save", "-s", action="store_true", help="Save multi-view grid preview PNG image to disk")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output image file path")
    parser.add_argument("--list", "-l", action="store_true", help="List all available patients in datasetmare")
    args = parser.parse_args()

    datasetmare_dir = args.dataset_dir
    if not os.path.exists(datasetmare_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        alt_dir = os.path.join(base_dir, DEFAULT_DATASET_DIR)
        if os.path.exists(alt_dir):
            datasetmare_dir = alt_dir

    if args.list:
        list_lidc_patients(datasetmare_dir)
        return

    xml_index = build_xml_index(datasetmare_dir)

    try:
        volume, slices_info, matched_xml, patient_id = load_patient_dicom_and_xml(
            args.patient, datasetmare_dir=datasetmare_dir, xml_index=xml_index
        )
        print(f"\nLoaded Patient: {patient_id}")
        print(f"Volume Shape: {volume.shape} (HU range: {volume.min():.1f} to {volume.max():.1f})")
        print(f"Matched Expert XML Annotation File: {matched_xml if matched_xml else 'None Found'}")

        masks_by_session, consensus_mask, attributes = parse_lidc_xml(matched_xml, slices_info)

        print(f"Parsed {len(masks_by_session)} Radiologist Sessions.")
        if consensus_mask is not None:
            consensus_voxels = np.sum(consensus_mask > 0)
            print(f"Expert Consensus Nodule Voxels: {int(consensus_voxels)}")

        if attributes:
            print("\nExpert Nodule Characteristics:")
            for attr in attributes[:5]:
                mal = attr.get('malignancy', 'N/A')
                sub = attr.get('subtlety', 'N/A')
                sph = attr.get('sphericity', 'N/A')
                print(f"  Radiologist {attr['session']+1} Nodule {attr['nodule']+1}: Malignancy Rating={mal}/5 | Subtlety={sub} | Sphericity={sph}")

        out_name = args.out or f"{patient_id}_expert_preview.png"
        save_lidc_grid_preview(volume, consensus_mask, masks_by_session, output_path=out_name, patient_id=patient_id)

        if not args.save:
            print("\nLaunching Interactive GUI Viewer...")
            viewer = InteractiveLIDCViewer(volume, consensus_mask, masks_by_session, attributes, patient_id=patient_id)
            viewer.show()

    except Exception as e:
        print(f"Error visualizing patient: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
