"""Repack the preprocessed npz slices into a pre-resized uint8 memmap store.

The npz files are zlib-compressed and hold per-annotator `session_masks` that training
never reads, so decoding a sample is orders of magnitude slower than a memmap read. This
repacks them once into fixed-shape uint8 arrays the DataLoader can memory-map directly.

Precision: images are HU-windowed and normalised to [0,1] during preprocessing, so one
uint8 step is well under CT reconstruction noise.

Layout, one pair of files per patient:
    <pid>_img.npy   uint8   (Z, 256, 256)       bilinear-resized, x255
    <pid>_msk.npy   uint8   (Z * 8192,)         np.packbits of the (Z,256,256) bool mask
    store_index.csv                             pid, n_slices, cs_h, cs_w, split

Slices pack to exactly 8192 bytes each (256*256/8), so slice z lives at
msk[z*8192:(z+1)*8192] with no bit-offset arithmetic.

Usage:
    python3 preprocess/build_store.py                      # train + val
    python3 preprocess/build_store.py --splits train val test
    python3 preprocess/build_store.py --workers 6
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DEFAULT_MANIFEST = "preprocessed_data/dataset_manifest.csv"
DEFAULT_OUT = "preprocessed_data_u8"
DEFAULT_SPLITS = ["train", "val"]
DEFAULT_WORKERS = 6
DEFAULT_CHUNK = 64

BYTES_PER_MASK = 256 * 256 // 8


def build_patient(job):
    pid, paths, out_dir, chunk = job
    out_dir = Path(out_dir)
    img_p = out_dir / f"{pid}_img.npy"
    msk_p = out_dir / f"{pid}_msk.npy"
    if img_p.exists() and msk_p.exists():
        return pid, 0, "skip", None

    Z = len(paths)
    with np.load(paths[0]) as f:
        cs = json.loads(str(f["spatial_meta"]))["cropped_shape"]
    cs_h, cs_w = float(cs[0]), float(cs[1])

    # tmp names must still end in .npy or np.save appends the suffix itself
    tmp_i = out_dir / f"{pid}_img.tmp.npy"
    tmp_m = out_dir / f"{pid}_msk.tmp.npy"
    arr = np.lib.format.open_memmap(tmp_i, mode="w+", dtype=np.uint8, shape=(Z, 256, 256))
    packed = np.empty(Z * BYTES_PER_MASK, dtype=np.uint8)

    for lo in range(0, Z, chunk):
        hi = min(lo + chunk, Z)
        ims, mks = [], []
        for p in paths[lo:hi]:
            with np.load(p) as f:
                ims.append(f["image"][0].astype(np.float32))
                mks.append(f["mask"][0])
        im = torch.from_numpy(np.stack(ims)).unsqueeze(1)
        mk = torch.from_numpy(np.stack(mks).astype(np.float32)).unsqueeze(1)
        # identical to the Resized transform the training pipeline applies
        im = F.interpolate(im, size=(256, 256), mode="bilinear", align_corners=False)[:, 0].numpy()
        mk = F.interpolate(mk, size=(256, 256), mode="nearest")[:, 0].numpy() > 0.5
        arr[lo:hi] = np.clip(np.rint(im * 255.0), 0, 255).astype(np.uint8)
        for k in range(hi - lo):
            packed[(lo + k) * BYTES_PER_MASK:(lo + k + 1) * BYTES_PER_MASK] = np.packbits(mk[k])

    arr.flush()
    del arr
    np.save(tmp_m, packed)
    os.replace(tmp_i, img_p)          # atomic: a crash leaves no half-written patient
    os.replace(tmp_m, msk_p)
    return pid, Z, "ok", (cs_h, cs_w)



def main():
    ap = argparse.ArgumentParser(
        description="Repack preprocessed npz slices into a uint8 memmap store")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help=f"Path to the dataset manifest CSV (default: {DEFAULT_MANIFEST})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Directory to write the store into (default: {DEFAULT_OUT})")
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                    help=f"Dataset splits to build (default: {' '.join(DEFAULT_SPLITS)})")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Parallel worker processes (default: {DEFAULT_WORKERS})")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                    help=f"Slices decoded at once; bounds worker memory on long scans "
                         f"(default: {DEFAULT_CHUNK})")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.manifest)
    df["filepath"] = df.filepath.str.replace("\\", "/", regex=False)
    df = df[df.split.isin(a.splits)].sort_values(["patient_id", "slice_idx"])

    jobs = []
    for pid, g in df.groupby("patient_id", sort=True):
        jobs.append((pid, list(g.filepath), str(out), a.chunk))
    split_of = df.groupby("patient_id").split.first().to_dict()

    print(f"store -> {out}")
    print(f"{len(jobs)} patients, {len(df):,} slices, splits={a.splits}, {a.workers} workers")
    est = len(df) * (256 * 256 + BYTES_PER_MASK) / 2 ** 30
    print(f"expected size ~{est:.1f} GiB\n")

    rows, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_patient, j): j[0] for j in jobs}
        for fu in as_completed(futs):
            pid, Z, status, cs = fu.result()
            done += 1
            if status == "skip":
                g = df[df.patient_id == pid]
                with np.load(g.filepath.iloc[0]) as f:
                    c = json.loads(str(f["spatial_meta"]))["cropped_shape"]
                rows.append((pid, len(g), float(c[0]), float(c[1]), split_of[pid]))
            else:
                rows.append((pid, Z, cs[0], cs[1], split_of[pid]))
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)}  {el:6.0f}s  "
                      f"eta {el/done*(len(jobs)-done):6.0f}s", flush=True)

    idx = pd.DataFrame(rows, columns=["patient_id", "n_slices", "cs_h", "cs_w", "split"])
    # Merge, never replace. Building one split at a time is normal; writing only this run's
    # rows would drop the others from the index while their .npy files sat on disk, and the
    # trainer would silently fall back to npz.
    prev_path = out / "store_index.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        keep = prev[~prev.patient_id.isin(set(idx.patient_id))]
        if len(keep):
            print(f"merging {len(keep)} existing index rows with {len(idx)} from this run")
        idx = pd.concat([keep, idx], ignore_index=True)
    idx.sort_values("patient_id").to_csv(prev_path, index=False)
    total = sum(os.path.getsize(out / f) for f in os.listdir(out))
    print(f"\ndone in {time.time()-t0:.0f}s | {total/2**30:.2f} GiB | "
          f"index -> {prev_path}")



if __name__ == "__main__":
    main()
