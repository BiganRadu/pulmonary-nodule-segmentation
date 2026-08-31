# Standard library
import os
import sys
import time
import json
import random
import argparse
import math
import multiprocessing
from collections import OrderedDict

# Windows-specific environment configuration
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch_lib_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib_dir):
        if torch_lib_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = torch_lib_dir + ";" + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(torch_lib_dir)
            except Exception:
                pass

# PyTorch & MONAI (Must be imported before pandas/scipy on Windows to prevent WinError 1114 DLL conflicts)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

# Each worker sends tensors to the main process through a queue. The default
# 'file_descriptor' strategy allocates one fd per tensor in flight, which stacks on
# top of the memmap fds below and exhausts RLIMIT_NOFILE on fast models.
torch.multiprocessing.set_sharing_strategy('file_system')

import monai
from monai.networks.nets import UNet, AttentionUnet, SegResNet
from monai.losses import DiceFocalLoss, DiceCELoss, TverskyLoss, FocalLoss, DiceLoss
from monai.transforms import (
    Compose,
    Resized,
    RandRotated,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped
)

# 3rd party
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import label

# Shared with evaluation/ so the gate applied during training is the same code
# that produces the reported metrics.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.postprocess import remove_small_objects_3d, component_elongation

# Default Configuration Constants
DEFAULT_MANIFEST = "preprocessed_data/dataset_manifest.csv"
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_MIN_LR = 1e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 8
DEFAULT_NEG_RATIO = 1.5
DEFAULT_LOSS = "dice_focal"
DEFAULT_MODEL_TYPE = "unet"
DEFAULT_SAVE_PATH = "models/unet_2d/unet_2d.pth"
DEFAULT_SEED = 42
DEFAULT_STORE_DIR = "preprocessed_data_u8"
DEFAULT_WARMUP_EPOCHS = 3
# fp16 has a narrow range (max 65504). An architecture whose activations grow large can
# overflow to Inf, which normalisation then turns into NaN, and every batch is skipped --
# training stops without an error. bf16 has fp32's exponent range and cannot overflow; it
# needs no GradScaler.
DEFAULT_AMP_DTYPE = "fp16"
# Weight on the positive class, for rescuing a run that has collapsed to predicting nothing.
# 0 disables and the loss chosen by --loss is used instead.
DEFAULT_POS_WEIGHT = 0.0
DEFAULT_EMA_DECAY = 0.999
DEFAULT_EMA_WARMUP = 10.0
DEFAULT_DICE_SMOOTH = 1.0
DEFAULT_NODULE_SIZES = "preprocessed_data/slice_nodule_size.csv"
DEFAULT_SIZE_ALPHA = 0.5
DEFAULT_SIZE_CAP = 4.0
# Probability threshold for the validation gate. Also part of the checkpoint-selection
# criterion, so it decides which epoch is kept, not just what gets reported.
DEFAULT_VAL_THRESHOLD = 0.5
# The rest of the gate is SWEPT at validation rather than fixed. The optimum is strongly
# model-dependent -- a model whose probabilities are less peaked needs a much lower peak
# threshold -- so a hardcoded value ranks epochs by how well they happen to match it rather
# than by how good they are. Labelling the connected components depends only on the
# threshold, so once they exist each combination below is arithmetic over per-component
# attributes and the whole grid costs about one extra labelling pass.
# A 0 disables that stage, matching remove_small_objects_3d.
VAL_SWEEP_MIN_VOXELS = (0, 15, 35, 70)
VAL_SWEEP_PEAK = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.998, 0.999, 0.9993, 0.9997)
VAL_SWEEP_ELONG = (0.0, 4.0, 3.0, 2.5, 2.0, 1.8)


def set_seed(seed=42):
    """
    Sets random seed for Python random, NumPy, and PyTorch (CPU & CUDA)
    for reproducible data shuffling and initial model weight generation,
    without forcing MONAI CUDNN determinism lock.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class DicePosWeightedBCELoss(torch.nn.Module):
    """
    Dice + BCE where the positive class carries `pos_weight`.

    This addresses a failure the `smooth` constant cannot touch. When a model collapses to
    predicting nothing, the gradient is not killed by sigmoid saturation as it first appears
    -- it is killed by the REDUCTION. BCE's derivative per foreground voxel is exactly
    (p - t) = -1, but the mean is taken over every voxel in the batch when only a tiny
    fraction are foreground, so almost none of that signal survives. Changing the smoothing
    constant does not help, because it barely moves the Dice term's gradient there either.
    Weighting the positive class restores the signal.

    pos_weight is aggressive by design and will over-fire at first; it is a way out of a
    collapse, not a default. Full class balance would be roughly 1 / (foreground fraction).
    Prefer a normal loss unless a run is actually stuck.
    """

    def __init__(self, pos_weight, smooth=1.0):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, squared_pred=True,
                             smooth_nr=smooth, smooth_dr=smooth)
        self.register_buffer("pw", torch.tensor(float(pos_weight)))

    def forward(self, logits, target):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pw.to(logits.device))
        return self.dice(logits, target) + bce



def get_loss_function(loss_name, smooth=DEFAULT_DICE_SMOOTH, pos_weight=0.0):
    """
    Returns configured MONAI loss function instance based on loss_name.
    Supported: 'dice_focal', 'dice_ce', 'tversky'

    `smooth` sets smooth_nr and smooth_dr, and it matters far more than a smoothing
    constant usually does, because ~60% of training samples have an EMPTY mask.

    For an empty target the squared-pred Dice term collapses to

        loss = 1 - smooth / (sum p^2 + smooth)

    so `smooth` alone decides where the loss starts responding. At a very small smooth the
    loss is pinned at 1.0 with a numerically zero gradient until sum p^2 falls below it,
    which over 65,536 pixels means every pixel driven to a strongly negative logit -- a wide
    dead zone followed by a cliff, and a run can spend most of its epochs crawling across it.

    At smooth = 1.0 the gradient peaks where sum p^2 ~ 1, i.e. around a single confident
    false pixel: the loss keeps pushing until essentially no pixel on an empty slice is
    confident, and stops there. Positive samples are unaffected, since +1 in numerator and
    denominator is negligible against a few hundred ground-truth voxels.

    NOTE: loss VALUES are not comparable across different `smooth` settings. The same model
    can report very different losses under two settings.
    """
    name = loss_name.lower().replace("-", "_")
    if pos_weight and pos_weight > 0:
        return DicePosWeightedBCELoss(pos_weight, smooth=smooth)
    if name in ["dice_ce", "dicece"]:
        return DiceCELoss(sigmoid=True, squared_pred=True, smooth_nr=smooth, smooth_dr=smooth)
    elif name in ["tversky", "focal_tversky", "focaltversky"]:
        class TverskyFocalLoss(torch.nn.Module):
            def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=smooth):
                super().__init__()
                self.tversky = TverskyLoss(sigmoid=True, alpha=alpha, beta=beta, smooth_nr=smooth, smooth_dr=smooth)
                self.focal = FocalLoss(include_background=True, to_onehot_y=False, gamma=gamma)

            def forward(self, pred, target):
                return self.tversky(pred, target) + self.focal(pred, target)

        return TverskyFocalLoss(alpha=0.3, beta=0.7, gamma=2.0, smooth=smooth)
    elif name in ["dice_focal", "dicefocal"]:
        return DiceFocalLoss(sigmoid=True, squared_pred=True, gamma=2.0,
                             smooth_nr=smooth, smooth_dr=smooth)
    else:
        raise ValueError(f"Unsupported loss function: '{loss_name}'. Choose from ['dice_focal', 'dice_ce', 'tversky']")




def get_model(model_type, in_channels=1, channels=None, strides=None):
    """
    Instantiates and returns (model, model_kwargs) based on model_type.
    Supported model types: 'unet', 'attention_unet', 'segresnet'
    """
    m_type = model_type.lower().replace("-", "_")
    if m_type in ["unet"]:
        model_kwargs = {
            "spatial_dims": 2,
            "in_channels": in_channels,
            "out_channels": 1,
            "channels": tuple(channels) if channels else (16, 32, 64, 128, 256),
            "strides": tuple(strides) if strides else (2, 2, 2, 2),
            "num_res_units": 2
        }
        model = UNet(**model_kwargs)
    elif m_type in ["attention_unet", "attentionunet", "att_unet", "attunet"]:
        model_kwargs = {
            "spatial_dims": 2,
            "in_channels": in_channels,
            "out_channels": 1,
            "channels": tuple(channels) if channels else (16, 32, 64, 128, 256),
            "strides": tuple(strides) if strides else (2, 2, 2, 2)
        }
        model = AttentionUnet(**model_kwargs)
    elif m_type in ["segresnet", "seg_resnet"]:
        model_kwargs = {
            "spatial_dims": 2,
            "in_channels": in_channels,
            "out_channels": 1,
            "init_filters": 16,
            "blocks_down": (1, 2, 2, 4),
            "blocks_up": (1, 1, 1)
        }
        model = SegResNet(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'. "
                         f"Choose from ['unet', 'attention_unet', 'segresnet']")

    return model, model_kwargs


def worker_init_fn(worker_id):
    """
    Per-worker initialisation for the DataLoaders.

    Seeds Python/NumPy AND the MONAI transform's own RandomState. MONAI random
    transforms each hold a private np.random.RandomState created in the parent process;
    seeding the global np.random does not touch it, so without this call every worker
    replays the identical augmentation stream and every epoch repeats the previous one.

    Also clears the memmap cache: file handles must never be shared across processes.
    """
    info = torch.utils.data.get_worker_info()
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    if info is not None:
        ds = info.dataset
        tf = getattr(ds, "transform", None)
        if tf is not None and hasattr(tf, "set_random_state"):
            tf.set_random_state(seed=worker_seed)
        if hasattr(ds, "_mm"):
            ds._mm = OrderedDict()


class LIDC2DDataset(Dataset):
    """
    PyTorch 2D Dataset for preprocessed LIDC-IDRI slices.
    Loads a single Z-slice [z] as a 1-channel input tensor (1, H, W).
    Target mask is that same slice's mask (1, H, W).

    Indexes the whole split. Which samples an epoch actually visits is decided by
    NegRatioSampler, so DataLoader workers can stay alive between epochs.

    Two backends:
      * uint8 memmap store (`store_dir`) -- pre-resized to 256x256, ~0.07 ms per sample
      * the original npz files            -- ~8 ms per sample, used when no store exists

    The index is four numpy arrays and one list of patient ids, rather than the full manifest
    DataFrame plus a dict of path strings -- fork() copies those into every worker, and at
    high worker counts that alone can exhaust system memory.
    """
    BYTES_PER_MASK = 256 * 256 // 8

    def __init__(self, manifest_csv, split="train", transform=None, neg_ratio=1.5, seed=42,
                 store_dir=None, nodule_size_csv=None):
        df = pd.read_csv(manifest_csv)
        df = df.sort_values(["patient_id", "slice_idx"]).reset_index(drop=True)

        self.pids = df["patient_id"].unique().tolist()
        pmap = {p: i for i, p in enumerate(self.pids)}
        pat_all = df["patient_id"].map(pmap).values.astype(np.int32)
        # slices per patient across ALL splits: 3D reconstruction at validation needs the
        # true volume length, not the length of this split
        self.n_slices = np.bincount(pat_all, minlength=len(self.pids)).astype(np.int32)

        sel = np.flatnonzero(df["split"].values == split)
        self.pat_i = pat_all[sel]
        self.z = df["slice_idx"].values[sel].astype(np.int32)
        self.has_tumor = df["has_tumor"].values[sel] > 0

        if split != "train" and neg_ratio <= 0:      # positives-only evaluation
            k = np.flatnonzero(self.has_tumor)
            self.pat_i, self.z, self.has_tumor = self.pat_i[k], self.z[k], self.has_tumor[k]

        self.split = split
        self.transform = transform
        self.neg_ratio = neg_ratio
        self.seed = seed
        self.root = os.path.dirname(os.path.dirname(
            str(df["filepath"].iloc[0]).replace("\\", "/"))) or "preprocessed_data"
        del df

        # in-plane extent in mm per patient, for the millimetre spacing used by the
        # surface-distance metrics
        self.cs_h = np.ones(len(self.pids), dtype=np.float32) * 256.0
        self.cs_w = np.ones(len(self.pids), dtype=np.float32) * 256.0
        self.store_dir = None
        if store_dir and os.path.isfile(os.path.join(store_dir, "store_index.csv")):
            idx = pd.read_csv(os.path.join(store_dir, "store_index.csv"))
            have = set(idx["patient_id"])
            need = {self.pids[i] for i in np.unique(self.pat_i)}
            if need <= have:
                self.store_dir = store_dir
                for pid, h, w in zip(idx["patient_id"], idx["cs_h"], idx["cs_w"]):
                    if pid in pmap:
                        self.cs_h[pmap[pid]] = h
                        self.cs_w[pmap[pid]] = w
            else:
                print(f"[store] {len(need - have)} patients of split '{split}' are missing "
                      f"from {store_dir}; falling back to npz")
        if self.store_dir is None:
            for i in np.unique(self.pat_i):
                pid = self.pids[i]
                with np.load(self._npz_path(i, 0)) as f:
                    cs = json.loads(str(f["spatial_meta"]))["cropped_shape"]
                self.cs_h[i], self.cs_w[i] = float(cs[0]), float(cs[1])

        # Per-slice volume of the ground-truth nodule that slice belongs to, for
        # NegRatioSampler's size-aware repetition. Zero where unknown or tumour-free, which
        # makes the repetition a no-op -- so a missing file simply disables the feature.
        self.nodule_vox = np.zeros(len(self.pat_i), dtype=np.float32)
        if nodule_size_csv and os.path.isfile(nodule_size_csv):
            ns = pd.read_csv(nodule_size_csv)
            ns = ns[ns["pid"].isin(pmap)]
            if len(ns):
                k_ds = self.pat_i.astype(np.int64) * 100000 + self.z.astype(np.int64)
                k_ns = ns["pid"].map(pmap).values.astype(np.int64) * 100000 + ns["z"].values
                o = np.argsort(k_ns)
                ks, vs = k_ns[o], ns["nodule_vox"].values[o].astype(np.float32)
                j = np.clip(np.searchsorted(ks, k_ds), 0, max(len(ks) - 1, 0))
                self.nodule_vox = np.where(ks[j] == k_ds, vs[j], 0.0).astype(np.float32)
                n_hit = int((self.nodule_vox > 0).sum())
                print(f"[nodule sizes] {n_hit:,} of {int(self.has_tumor.sum()):,} tumour slices "
                      f"in split '{split}' matched from {nodule_size_csv}")

        self._mm = OrderedDict()

    # ------------------------------------------------------------------ backends
    def _npz_path(self, pat_i, z):
        pid = self.pids[pat_i]
        return f"{self.root}/{pid}/{pid}_slice{int(z):03d}.npz"

    MM_CACHE = 192

    def _store(self, pat_i):
        """Open (and cache) the memmaps for one patient.

        numpy holds an open file descriptor for the lifetime of each memmap, so an
        unbounded cache costs 2 fds per patient per worker -- over 1800 on this dataset,
        which trips RLIMIT_NOFILE. Evict least-recently-used instead. `_read` copies out
        of the memmap before returning, so nothing survives eviction.
        """
        mm = self._mm.get(pat_i)
        if mm is not None:
            self._mm.move_to_end(pat_i)
            return mm
        pid = self.pids[pat_i]
        mm = (np.load(f"{self.store_dir}/{pid}_img.npy", mmap_mode="r"),
              np.load(f"{self.store_dir}/{pid}_msk.npy", mmap_mode="r"))
        self._mm[pat_i] = mm
        while len(self._mm) > self.MM_CACHE:
            _, old = self._mm.popitem(last=False)
            for arr in old:
                base = getattr(arr, "_mmap", None)
                if base is not None:
                    base.close()
        return mm

    def _read(self, pat_i, z):
        if self.store_dir is not None:
            img, msk = self._store(pat_i)
            x = img[z:z + 1].astype(np.float32) / 255.0
            b = self.BYTES_PER_MASK
            m = np.unpackbits(msk[z * b:(z + 1) * b]).reshape(1, 256, 256).astype(np.float32)
            return x, m
        with np.load(self._npz_path(pat_i, z)) as f:
            return f["image"].astype(np.float32), f["mask"].astype(np.float32)

    def __len__(self):
        return len(self.pat_i)

    def __getitem__(self, idx):
        pat_i = int(self.pat_i[idx])
        z = int(self.z[idx])
        last = int(self.n_slices[pat_i]) - 1

        image, mask = self._read(pat_i, z)
        sample = {"image": torch.from_numpy(np.ascontiguousarray(image)),
                  "mask": torch.from_numpy(np.ascontiguousarray(mask))}
        if self.transform:
            sample = self.transform(sample)
        cropped_shape = [float(self.cs_h[pat_i]), float(self.cs_w[pat_i]), float(last + 1)]
        return sample["image"], sample["mask"], self.pids[pat_i], z, cropped_shape


class NegRatioSampler(Sampler):
    """
    Draws every positive slice plus a fresh random subset of negatives each epoch.

    This lived inside the Dataset before, which forced persistent_workers=False so that
    re-forked workers would pick up the new selection. As a Sampler it is consulted from
    the main process each epoch, so workers can stay alive -- and the resampling itself
    is vectorised (~10 ms instead of 1.4 s of GPU-idle time per epoch).
    """
    def __init__(self, dataset, neg_ratio=1.5, seed=42, clean_base=20,
                 size_alpha=0.0, size_cap=4.0):
        self.ds = dataset
        self.neg_ratio = neg_ratio
        self.seed = seed
        self.clean_base = clean_base
        self.epoch = 0

        # Size-aware repetition of positives. A nodule contributes one training sample per
        # slice it spans, and Z-extent tracks volume closely -- so the smallest nodules make
        # up a larger share of all nodules than they do of the positive slices,
        # while those over 1000 supply 33%. Repeating small-nodule slices evens that out.
        # Augmentation is re-randomised per sample, so a repeat is a new view, not a copy.
        self.reps = np.ones(len(dataset.pat_i), dtype=np.int32)
        nv = getattr(dataset, "nodule_vox", None)
        if size_alpha > 0 and nv is not None and (nv > 0).any():
            has = dataset.has_tumor & (nv > 0)
            med = float(np.median(nv[has]))
            r = np.clip((med / np.maximum(nv, 1.0)) ** size_alpha, 1.0, size_cap)
            self.reps[has] = np.rint(r[has]).astype(np.int32)
            print(f"[sampler] size-aware repeats: alpha={size_alpha} cap={size_cap} "
                  f"median nodule {med:.0f} vox | positives {int(has.sum()):,} -> "
                  f"{int(self.reps[has].sum()):,} samples "
                  f"({self.reps[has].sum()/max(int(has.sum()),1):.2f}x)")
        n_p = len(dataset.pids)
        order = np.arange(n_p)
        self.starts = np.searchsorted(dataset.pat_i, order, side="left")
        self.ends = np.searchsorted(dataset.pat_i, order, side="right")
        # exact and epoch-independent: the count depends only on how many positives
        # each patient has, never on which negatives get drawn
        total = 0
        for a, b in zip(self.starts, self.ends):
            if b <= a:
                continue
            m = dataset.has_tumor[a:b]
            # count repeats, so __len__ stays exact and epoch-independent
            n_pos, n_neg = int(self.reps[a:b][m].sum()), int((~m).sum())
            total += n_pos + min(n_neg, self._target(n_pos))
        self._len = total

    def _target(self, n_pos):
        if n_pos > 0:
            return int(np.ceil(self.neg_ratio * n_pos))
        if self.neg_ratio <= 0:
            return 0
        return int(round(self.clean_base * self.neg_ratio / 1.5))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _build(self):
        rng = np.random.RandomState((self.seed * 100003 + self.epoch) % (2 ** 31 - 1))
        parts = []
        ht = self.ds.has_tumor
        for a, b in zip(self.starts, self.ends):
            if b <= a:
                continue
            m = ht[a:b]
            pos = np.flatnonzero(m) + a
            neg = np.flatnonzero(~m) + a
            pos = np.repeat(pos, self.reps[pos])   # small nodules appear more often
            parts.append(pos)
            t = self._target(len(pos))             # negatives scale, so the mix is unchanged
            if len(neg) and t > 0:
                parts.append(rng.choice(neg, size=min(len(neg), t), replace=False))
        out = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        rng.shuffle(out)
        return out

    def __iter__(self):
        return iter(self._build().tolist())

    def __len__(self):
        return self._len


def get_transforms(no_augmentations=False, resize=True):
    """
    Returns MONAI transform pipelines for 2D inputs.

    `resize=False` drops the 256x256 Resized step, for when the uint8 store already
    delivers pre-resized slices. Augmentations are unchanged either way, so a model
    trained off the store sees the same distribution as one trained off npz.
    """
    pre = [Resized(keys=["image", "mask"], spatial_size=(256, 256),
                   mode=["bilinear", "nearest"])] if resize else []

    if no_augmentations:
        train_transforms = Compose(pre + [EnsureTyped(keys=["image", "mask"])])
    else:
        train_transforms = Compose(pre + [
            RandRotated(
                keys=["image", "mask"],
                range_x=0.26,                  # +/-15 degrees random rotation
                mode=["bilinear", "nearest"],
                prob=0.5
            ),
            RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.03),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
            EnsureTyped(keys=["image", "mask"])
        ])

    val_transforms = Compose(pre + [EnsureTyped(keys=["image", "mask"])])
    return train_transforms, val_transforms


def plot_training_history(history, save_path="models/unet_2d/training_history.png"):
    """
    Four panels aimed at a 3D-Dice objective: loss, the gated vs raw 3D Dice that
    selection depends on, the false-positive picture that drives 2D all-slice Dice, and
    detection against tumour-slice Dice as the guard metric.
    """
    epochs = history.get('epoch', [])
    if not epochs:
        return

    matplotlib.use('Agg')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color='royalblue', linewidth=2, label='Train Loss')
    ax.set_title('Training Loss', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend()

    ax = axes[0, 1]
    if 'val_3d_gated' in history:
        ax.plot(epochs, history['val_3d_gated'], color='crimson', linewidth=2.5,
                label='3D Dice, gated  (selection metric)')
    if 'val_3d_dice' in history:
        ax.plot(epochs, history['val_3d_dice'], color='grey', linestyle='--', linewidth=1.5,
                label='3D Dice, raw')
    ax.set_title('3D Patient Dice', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Dice [0-1]')
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend(loc='lower right', fontsize=9)

    ax = axes[1, 0]
    if 'val_dice_all' in history:
        ax.plot(epochs, history['val_dice_all'], color='darkorange', linewidth=2,
                label='2D Dice, all slices')
    if 'val_fa' in history:
        ax.plot(epochs, history['val_fa'], color='firebrick', linestyle='--', linewidth=1.5,
                label='False-alarm rate on tumour-free slices')
    ax.set_title('False Positives', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Rate [0-1]')
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend(loc='center right', fontsize=9)
    if 'val_fp_comp' in history:
        ax2 = ax.twinx()
        ax2.plot(epochs, history['val_fp_comp'], color='seagreen', linestyle=':', linewidth=1.5)
        ax2.set_ylabel('False components per scan', color='seagreen', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='seagreen')

    ax = axes[1, 1]
    if 'val_dice' in history:
        ax.plot(epochs, history['val_dice'], color='darkorange', linewidth=2,
                label='2D Dice, tumour slices (guard)')
    if 'val_sens' in history:
        ax.plot(epochs, history['val_sens'], color='teal', linestyle='--', linewidth=1.5,
                label='Sensitivity')
    if 'val_prec' in history:
        ax.plot(epochs, history['val_prec'], color='purple', linestyle='--', linewidth=1.5,
                label='Precision')
    if 'val_detect' in history:
        ax.plot(epochs, history['val_detect'], color='black', linewidth=1.5,
                label='Nodules detected after the gate')
    ax.set_title('Segmentation Quality & Detection', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Score [0-1]')
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_epoch(model, loader, optimizer, loss_fn, device, epoch, total_epochs, scaler=None, amp_dtype=torch.float16, ema=None):
    """
    Trains MONAI 2D UNet for 1 epoch using PyTorch AMP (BF16 / FP16) precision.
    """
    model.train()
    running_loss = 0.0
    n_skipped = 0
    seen = 0
    start_time = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train 2D]", leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar, 1):
        images, masks = batch[0], batch[1]  # batch[2:] = pid, slice_idx, cropped_shape (unused in training)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(images)
                loss = loss_fn(logits, masks)

            if torch.isnan(loss) or torch.isinf(loss):
                n_skipped += 1
                if n_skipped == 1:
                    print(f"\n  [warn] epoch {epoch}: non-finite loss, batch skipped. With "
                          f"amp_dtype=fp16 this usually means activation overflow -- try "
                          f"--amp_dtype bf16.", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if ema is not None:
            ema.update_parameters(model)
        loss_val = loss.item()
        running_loss += loss_val * images.size(0)
        seen += images.size(0)
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    # len(loader.dataset) is the whole split; with NegRatioSampler an epoch visits only a
    # subset of it, so normalise by the samples actually seen
    if seen == 0:
        raise SystemExit(
            f"epoch {epoch}: every batch produced a non-finite loss, so no weights were "
            f"updated. Reporting this as a loss of 0.0 would look like convergence; it is "
            f"the opposite. Almost always fp16 overflow -- rerun with --amp_dtype bf16.")
    if n_skipped:
        print(f"  [warn] epoch {epoch}: skipped {n_skipped} batches on non-finite loss", flush=True)
    epoch_loss = running_loss / seen
    elapsed = time.time() - start_time
    return epoch_loss, elapsed


def ema_multi_avg_fn_warmup(decay=0.999, warmup=10.0):
    """
    EMA update whose decay ramps in instead of starting at its final value.

    AveragedModel seeds the average with the weights present at the first update -- i.e.
    the random initialisation. At a fixed decay of 0.999 that init decays by 0.999 per step,
    so a substantial fraction of the average is still noise after the first epoch and several
    epochs are spent washing it out.

    Ramping as min(decay, (1 + n) / (warmup + n)) makes the average track the live weights
    almost exactly for the first few steps and reach `decay` after a few thousand, which
    is what timm and Mean Teacher do. `n` comes from AveragedModel.n_averaged, so it is
    part of the checkpoint and survives a resume.
    """
    @torch.no_grad()
    def ema_update(ema_params, cur_params, num_averaged):
        n = float(num_averaged.item()) if torch.is_tensor(num_averaged) else float(num_averaged)
        d = min(decay, (1.0 + n) / (float(warmup) + n))
        if torch.is_floating_point(ema_params[0]) or torch.is_complex(ema_params[0]):
            torch._foreach_lerp_(ema_params, cur_params, 1.0 - d)
        else:
            for pe, pm in zip(ema_params, cur_params):
                pe.copy_(pe * d + pm * (1.0 - d))
    return ema_update


def validate_epoch(model, loader, device, use_amp=True, amp_dtype=torch.float16, gate=None):
    """
    Validation for a 3D-Dice objective, with the post-processing gate swept rather than fixed.

    Reconstructs one patient at a time (the loader is shuffle=False and the dataset is
    ordered by (patient_id, slice_idx), so a patient's slices arrive contiguously) and
    scores it raw, then under every (min_voxels, peak, elongation) combination in
    VAL_SWEEP_*. The reported gated metrics are the ones from the best-scoring combination,
    and that is what checkpoint selection uses.

    3D Dice is reported under BOTH conventions, because they are not interchangeable:
      * dice_3d_gated      -- ALL patients. A nodule-free scan scores exactly 1.0 if the
                              prediction is empty and 0.0 otherwise. ~19% of val patients
                              are nodule-free, so this is the objective and the selection
                              metric.
      * dice_3d_gated_nod  -- nodule-bearing scans only. Informative, but blind to the
                              clean scans that a fifth of the objective rides on.
    """
    model.eval()
    ST = np.ones((3, 3, 3), dtype=bool)

    # flattened gate grid, evaluated for every patient
    CFG = [(mv, pk, el) for mv in VAL_SWEEP_MIN_VOXELS
           for pk in VAL_SWEEP_PEAK for el in VAL_SWEEP_ELONG]
    MV = np.array([c[0] for c in CFG], dtype=np.float64)[:, None]
    PK = np.array([c[1] for c in CFG], dtype=np.float64)[:, None]
    EL = np.array([c[2] for c in CFG], dtype=np.float64)[:, None]
    NC = len(CFG)
    acc_dice = np.zeros(NC)        # summed per-patient 3D Dice, every patient
    acc_dice_nod = np.zeros(NC)    # summed, nodule-bearing patients only
    acc_clean_ok = np.zeros(NC)    # nodule-free patients predicted empty
    acc_fp = np.zeros(NC)          # surviving components touching no ground truth
    acc_found = np.zeros(NC)       # ground-truth nodules hit by a surviving component

    sample_dices, sample_precisions, sample_sensitivities = [], [], []
    failures = 0
    total_pos_samples = 0

    n_slices = n_neg = n_neg_hit = 0
    dice_all_sum = 0.0
    d3_raw = []                            # nodule-bearing scans only
    d3_raw_all = []                        # every scan, clean ones included
    n_clean = 0
    n_nod_total = 0
    n_patients = 0

    cur = {"pid": None, "prob": [], "gt": [], "cs": None}

    def finalize():
        nonlocal n_nod_total, n_patients, n_clean
        nonlocal acc_dice, acc_dice_nod, acc_clean_ok, acc_fp, acc_found
        if cur["pid"] is None:
            return
        prob = np.stack(cur["prob"], axis=-1)
        gt = np.stack(cur["gt"], axis=-1)
        raw = prob > (gate["threshold"] if gate else 0.5)
        n_patients += 1

        g_sum = int(gt.sum())
        p_sum = int(raw.sum())
        d_raw = 1.0 if (g_sum + p_sum) == 0 else 2.0 * int((raw & gt).sum()) / (g_sum + p_sum)
        d3_raw_all.append(d_raw)
        if g_sum > 0:
            d3_raw.append(d_raw)

        if gate is not None:
            cs_h, cs_w = cur["cs"]
            spacing = (cs_h / 256.0, cs_w / 256.0, 1.0)

            # ---- one labelling pass, then per-component attributes ----
            lab, n = label(raw, structure=ST) if p_sum else (None, 0)
            if n:
                flat = lab.ravel()
                fg = np.flatnonzero(flat)
                cid = flat[fg]
                size = np.bincount(cid, minlength=n + 1)[1:].astype(np.float64)
                peak = np.zeros(n + 1, dtype=np.float32)
                np.maximum.at(peak, cid, prob.ravel()[fg].astype(np.float32))
                peak = peak[1:].astype(np.float64)
                elong = component_elongation(lab, n, spacing)[1:]
                gtf = gt.ravel()[fg]
                inter = np.bincount(cid, weights=gtf.astype(np.float64),
                                    minlength=n + 1)[1:]

                # which ground-truth nodule each component touches
                if g_sum > 0:
                    gl, ng = label(gt, structure=ST)
                    n_nod_total += ng
                    hit = np.zeros((n, ng), dtype=bool)
                    glf = gl.ravel()[fg]
                    m = glf > 0
                    if m.any():
                        hit[cid[m] - 1, glf[m] - 1] = True
                else:
                    hit = np.zeros((n, 0), dtype=bool)

                # ---- every gate at once: pure arithmetic over the components ----
                # a gate stage is inactive at 0, matching remove_small_objects_3d
                keep = (((MV <= 0) | (size[None, :] >= MV))
                        & ((PK <= 0) | (peak[None, :] >= PK))
                        & ((EL <= 0) | (elong[None, :] <= EL)))
                pred = keep @ size
                tp = keep @ inter
                acc_fp += keep @ (inter == 0).astype(np.float64)
                found = ((keep.astype(np.int64) @ hit.astype(np.int64)) > 0).sum(1)
            else:
                if g_sum > 0:
                    _, ng = label(gt, structure=ST)
                    n_nod_total += ng
                pred = np.zeros(NC)
                tp = np.zeros(NC)
                found = np.zeros(NC)

            den = g_sum + pred
            d = np.where(den == 0, 1.0, 2.0 * tp / np.maximum(den, 1e-9))
            acc_dice += d
            if g_sum == 0:
                n_clean += 1
                acc_clean_ok += (pred == 0)
            else:
                acc_dice_nod += d
                acc_found += found
        cur["prob"] = []
        cur["gt"] = []

    pbar = tqdm(loader, desc="[Val]", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            images, masks, pids = batch[0], batch[1], batch[2]
            cropped_shapes = batch[4]
            images = images.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(images)
            else:
                logits = model(images)

            # cast before sigmoid, matching evaluation/evaluate_2_5d.py
            probs_np = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
            masks_np = masks[:, 0].numpy() > 0.5
            th = gate["threshold"] if gate else 0.5

            for b in range(probs_np.shape[0]):
                pid = pids[b]
                if pid != cur["pid"]:
                    finalize()
                    cur["pid"] = pid
                    cur["cs"] = (float(cropped_shapes[0][b]), float(cropped_shapes[1][b]))
                cur["prob"].append(probs_np[b])
                cur["gt"].append(masks_np[b])

                p_mask = probs_np[b] > th
                g_mask = masks_np[b]

                g_px = int(g_mask.sum())
                p_px = int(p_mask.sum())
                inter = int((p_mask & g_mask).sum())
                n_slices += 1
                dice_all_sum += 1.0 if (g_px + p_px) == 0 else (2.0 * inter) / (g_px + p_px)

                if g_px == 0:
                    n_neg += 1
                    n_neg_hit += int(p_px > 0)
                    continue

                total_pos_samples += 1
                dice = (2.0 * inter) / (g_px + p_px + 1e-8)
                sample_dices.append(dice)
                sample_precisions.append(inter / (p_px + 1e-8))
                sample_sensitivities.append(inter / (g_px + 1e-8))
                if dice < 0.1:
                    failures += 1
    finalize()

    mean = lambda v: float(np.mean(v)) if len(v) else 0.0
    metrics = {
        "dice": mean(sample_dices),
        "precision": mean(sample_precisions),
        "sensitivity": mean(sample_sensitivities),
        "failure_rate": float(failures / total_pos_samples) if total_pos_samples else 0.0,
        "dice_all": float(dice_all_sum / n_slices) if n_slices else 0.0,
        "false_alarm_rate": float(n_neg_hit / n_neg) if n_neg else 0.0,
        "val_3d_dice": mean(d3_raw),
        "dice_3d_raw_all": mean(d3_raw_all),
    }

    if gate is not None and n_patients:
        best = int(np.argmax(acc_dice))
        n_nod_pat = n_patients - n_clean
        metrics.update({
            # ALL-patient gated Dice at the best gate: the objective, and what selection uses
            "dice_3d_gated": float(acc_dice[best] / n_patients),
            "dice_3d_gated_nod": float(acc_dice_nod[best] / n_nod_pat) if n_nod_pat else 0.0,
            "clean_empty_rate": float(acc_clean_ok[best] / n_clean) if n_clean else 0.0,
            "fp_comp_per_scan": float(acc_fp[best] / n_patients),
            "detect_rate": float(acc_found[best] / n_nod_total) if n_nod_total else 0.0,
            "best_gate": CFG[best],
        })
    else:
        metrics.update({
            "dice_3d_gated": mean(d3_raw_all),
            "dice_3d_gated_nod": mean(d3_raw),
            "clean_empty_rate": 0.0,
            "fp_comp_per_scan": 0.0,
            "detect_rate": 0.0,
            "best_gate": None,
        })
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MONAI 2D UNet Model on LIDC-IDRI Dataset")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help=f"Path to master manifest CSV (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Total target training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--min_lr", type=float, default=DEFAULT_MIN_LR, help=f"Minimum learning rate for scheduler (default: {DEFAULT_MIN_LR})")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help=f"Weight decay for AdamW optimizer (default: {DEFAULT_WEIGHT_DECAY})")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help=f"DataLoader num_workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--neg_ratio", type=float, default=DEFAULT_NEG_RATIO, help=f"Ratio of negative slices to positive slices for dynamic epoch resampling (default: {DEFAULT_NEG_RATIO})")
    parser.add_argument("--loss", type=str, choices=["dice_focal", "dice_ce", "tversky", "focal_tversky"], default=DEFAULT_LOSS, help=f"Loss function to optimize: dice_focal, dice_ce, tversky, or focal_tversky (default: {DEFAULT_LOSS})")
    parser.add_argument("--model_type", "--model", type=str, choices=["unet", "attention_unet", "segresnet"], default=DEFAULT_MODEL_TYPE, help=f"Model architecture to train (default: {DEFAULT_MODEL_TYPE})")
    parser.add_argument("--no_transforms", "--no_aug", action="store_true", help="Disable random data augmentations during training (keeps 256x256 resizing only)")
    parser.add_argument("--save_path", "--model_path", type=str, default=DEFAULT_SAVE_PATH, help=f"Path to save best model checkpoint or output folder (default: {DEFAULT_SAVE_PATH})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for data sampling & initial weights (default: {DEFAULT_SEED})")
    parser.add_argument("--pos_weight", type=float, default=DEFAULT_POS_WEIGHT, help=f"Override the loss with Dice + BCE weighting the positive class by this factor. For rescuing a run that has collapsed to predicting nothing, where the gradient is destroyed by BCE's mean over every voxel when almost none are foreground -- something --dice_smooth cannot fix. 0 disables (default: {DEFAULT_POS_WEIGHT})")
    parser.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16", "fp32"], default=DEFAULT_AMP_DTYPE, help=f"Autocast precision. fp16 can overflow on architectures with large activations, after which every batch is skipped and training stops silently; bf16 has fp32's range and needs no GradScaler (default: {DEFAULT_AMP_DTYPE})")
    parser.add_argument("--warmup_epochs", type=int, default=DEFAULT_WARMUP_EPOCHS, help=f"Linear LR warmup before cosine annealing (default: {DEFAULT_WARMUP_EPOCHS})")
    parser.add_argument("--dice_smooth", type=float, default=DEFAULT_DICE_SMOOTH, help=f"smooth_nr/smooth_dr for the Dice term. MONAI defaults to 1e-5, which leaves a zero-gradient plateau on empty masks and stalls training for tens of epochs; see get_loss_function (default: {DEFAULT_DICE_SMOOTH})")
    parser.add_argument("--ema_warmup", type=float, default=DEFAULT_EMA_WARMUP, help=f"Steps over which the EMA decay ramps to --ema_decay; stops the random init from dominating the average for the first several epochs (default: {DEFAULT_EMA_WARMUP})")
    parser.add_argument("--ema_decay", type=float, default=DEFAULT_EMA_DECAY, help=f"Exponential moving average of weights, validated instead of the raw model. 0 disables (default: {DEFAULT_EMA_DECAY})")
    parser.add_argument("--nodule_sizes", type=str, default=DEFAULT_NODULE_SIZES, help=f"CSV of per-slice ground-truth nodule volume (pid,z,slice_px,nodule_vox), used only when --size_alpha > 0 (default: {DEFAULT_NODULE_SIZES})")
    parser.add_argument("--size_alpha", type=float, default=DEFAULT_SIZE_ALPHA, help=f"Repeat positive slices by clip((median_vox/nodule_vox)**alpha, 1, --size_cap), so slices holding small nodules are sampled more often. 0 disables; higher values trade a larger epoch for a bigger small-nodule share (default: {DEFAULT_SIZE_ALPHA})")
    parser.add_argument("--size_cap", type=float, default=DEFAULT_SIZE_CAP, help=f"Maximum repeat factor for --size_alpha (default: {DEFAULT_SIZE_CAP})")
    parser.add_argument("--val_threshold", type=float, default=DEFAULT_VAL_THRESHOLD, help=f"Probability threshold used by the validation gate (default: {DEFAULT_VAL_THRESHOLD})")
    parser.add_argument("--store_dir", type=str, default=DEFAULT_STORE_DIR, help=f"Pre-resized uint8 memmap store built by preprocess/build_store.py. Falls back to the npz files if absent (default: {DEFAULT_STORE_DIR})")
    parser.add_argument("--no_store", action="store_true", help="Ignore the uint8 store and read the original npz files")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")

    args = parser.parse_args()

    # Set seed for PyTorch, NumPy, and Python without forcing CUDNN determinism lock
    set_seed(args.seed)

    # Automatically derive default save directory based on chosen model_type if default path was used
    if args.save_path == DEFAULT_SAVE_PATH and args.model_type != DEFAULT_MODEL_TYPE:
        args.save_path = f"models/{args.model_type}_2d/{args.model_type}_2d.pth"

    # Automatically derive all output artifact paths from --save_path directory
    raw_save_path = os.path.normpath(args.save_path)
    if raw_save_path.endswith(".pth"):
        save_dir = os.path.dirname(raw_save_path) or "."
        best_model_path = raw_save_path
    else:
        save_dir = raw_save_path
        best_model_path = os.path.join(save_dir, f"{args.model_type}_2d.pth")

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pth")
    results_log = os.path.join(save_dir, "train.txt")
    history_plot_path = os.path.join(save_dir, "training_history.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                     "fp32": torch.float32}[args.amp_dtype]
        if amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise SystemExit("--amp_dtype bf16 requested but this GPU does not support it")
        # GradScaler exists to rescue fp16's narrow range; bf16 and fp32 do not need it
        scaler = torch.amp.GradScaler('cuda') if amp_dtype is torch.float16 else None
        print(f"Device: {device} | AMP Dtype: {amp_dtype} | GradScaler: {scaler is not None}")
    else:
        amp_dtype = torch.float32
        scaler = None

    store_dir = None if args.no_store else args.store_dir
    use_store = bool(store_dir) and os.path.isfile(os.path.join(store_dir, "store_index.csv"))
    # the store is already 256x256, so the Resized step would be wasted work
    train_transforms, val_transforms = get_transforms(no_augmentations=args.no_transforms,
                                                      resize=not use_store)

    train_dataset = LIDC2DDataset(args.manifest, split="train", transform=train_transforms,
                                  neg_ratio=args.neg_ratio, seed=args.seed, store_dir=store_dir,
                                  nodule_size_csv=args.nodule_sizes)
    val_dataset = LIDC2DDataset(args.manifest, split="val", transform=val_transforms,
                                neg_ratio=args.neg_ratio, seed=args.seed, store_dir=store_dir)
    train_sampler = NegRatioSampler(train_dataset, neg_ratio=args.neg_ratio, seed=args.seed,
                                    size_alpha=args.size_alpha, size_cap=args.size_cap)
    backend = f"uint8 store ({store_dir})" if train_dataset.store_dir else "npz"
    print(f"Data backend: {backend} | train {len(train_sampler):,} samples/epoch "
          f"(neg_ratio {args.neg_ratio}) | val {len(val_dataset):,} slices")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,     # negative resampling lives here, so workers can persist
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
        prefetch_factor=4 if args.num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
        prefetch_factor=4 if args.num_workers > 0 else None
    )

    model, model_kwargs = get_model(args.model_type, in_channels=1)
    model = model.to(device)

    loss_fn = get_loss_function(args.loss, smooth=args.dice_smooth, pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True if device.type == 'cuda' else False)

    start_epoch = 1
    best_gated_3d = -1.0
    history = {
        "epoch": [], "train_loss": [], "val_3d_dice": [], "val_3d_gated": [],
        "val_3d_gated_nod": [], "val_clean_ok": [],
        "val_dice": [], "val_dice_all": [], "val_prec": [], "val_sens": [],
        "val_fa": [], "val_fp_comp": [], "val_detect": [], "val_fail_rate": []
    }

    if args.warmup_epochs > 0:
        # ramp in, then cosine over the remaining epochs
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=args.min_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, cos], milestones=[args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # Weights EMA. Gated 3D Dice swings a lot epoch to epoch, so selecting on the raw
    # model is close to a coin flip; the averaged weights are what gets validated.
    ema_model = None
    if args.ema_decay and args.ema_decay > 0:
        # The multi_avg_fn form operates on the whole parameter list at once; passing it as
        # avg_fn calls it per-tensor and raises.
        ema_model = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=ema_multi_avg_fn_warmup(args.ema_decay, args.ema_warmup),
            use_buffers=True)

    VAL_GATE = dict(threshold=args.val_threshold)
    n_cfg = len(VAL_SWEEP_MIN_VOXELS) * len(VAL_SWEEP_PEAK) * len(VAL_SWEEP_ELONG)
    print(f"Validation gate: th={VAL_GATE['threshold']}, swept over {n_cfg} "
          f"(min_voxels, peak, elong) combinations; best one is reported and selected on "
          f"| ema={args.ema_decay} | warmup={args.warmup_epochs}")

    # Resume capability from checkpoint
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_gated_3d = checkpoint.get("best_gated_3d", -1.0)

        prev = checkpoint.get("history")
        if isinstance(prev, dict) and set(prev) == set(history):
            history = prev
        elif prev:
            print("[resume] checkpoint history has a different metric set; starting a fresh history")

        if "scaler_state_dict" in checkpoint and scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # SequentialLR does NOT round-trip through state_dict -- restoring it yields the
        # warmup learning rate again, an error that then persists for the rest of the run.
        # Fast-forwarding a freshly built scheduler reproduces the schedule exactly.
        for _ in range(start_epoch - 1):
            scheduler.step()
        # optimizer.load_state_dict() above replaced param_groups with the SAVED lr, which
        # the scheduler will not overwrite until its first step at the END of this epoch.
        # Without this the resumed epoch runs at whatever lr the source run happened to
        # stop at -- harmless when continuing the same schedule, wrong for a warm restart
        # at a different --lr.
        for g, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            g["lr"] = lr

        if ema_model is not None and checkpoint.get("ema_state_dict") is not None:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
        elif ema_model is not None:
            print("[resume] no EMA state in the checkpoint; the average restarts from the "
                  "current weights")
        print(f"[resume] continuing from epoch {start_epoch}, best gated 3D Dice so far "
              f"{best_gated_3d:.4f}, lr {optimizer.param_groups[0]['lr']:.2e}")

    # Initialize results log file header if starting fresh
    if not args.resume or not os.path.exists(results_log):
        with open(results_log, "a") as f:
            f.write(f"\n--- Training Session Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write("Epoch | Loss | 3DAll | 3DNod | 3DRaw | CleanOK | 2DAll | 2DTumor | FA | FPcomp | Detect | Precision | Sensitivity | FailRate | Gate\n")

    total_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)   # fresh negative draw, no worker respawn

        train_loss, train_time = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, args.epochs, scaler=scaler, amp_dtype=amp_dtype, ema=ema_model)
        eval_model = ema_model.module if ema_model is not None else model
        val_metrics = validate_epoch(eval_model, val_loader, device,
                                     use_amp=True, amp_dtype=amp_dtype, gate=VAL_GATE)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_3d_dice"].append(val_metrics["dice_3d_raw_all"])
        history["val_3d_gated"].append(val_metrics["dice_3d_gated"])
        history["val_3d_gated_nod"].append(val_metrics["dice_3d_gated_nod"])
        history["val_clean_ok"].append(val_metrics["clean_empty_rate"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_dice_all"].append(val_metrics["dice_all"])
        history["val_prec"].append(val_metrics["precision"])
        history["val_sens"].append(val_metrics["sensitivity"])
        history["val_fa"].append(val_metrics["false_alarm_rate"])
        history["val_fp_comp"].append(val_metrics["fp_comp_per_scan"])
        history["val_detect"].append(val_metrics["detect_rate"])
        history["val_fail_rate"].append(val_metrics["failure_rate"])

        # Select on ALL-patient gated 3D Dice -- the objective. The `dice > 0` guard is
        # not theoretical: selecting on 2D all-slice Dice once saved an untrained network,
        # because an empty prediction scores the empty-mask baseline and beats every
        # trained epoch.
        is_best = val_metrics["dice"] > 0.0 and val_metrics["dice_3d_gated"] > best_gated_3d
        if is_best:
            best_gated_3d = val_metrics["dice_3d_gated"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": (ema_model.module if ema_model is not None else model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_metrics["dice"],
                "val_3d_dice": val_metrics["val_3d_dice"],
                "dice_3d_gated": val_metrics["dice_3d_gated"],
                "val_gate": VAL_GATE,
                "val_metrics": val_metrics,
                "model_kwargs": model_kwargs,
                "in_slices": 1,
                "model_type": args.model_type
            }, best_model_path)
        # last.pth always: the val-optimal checkpoint does not always transfer to test
        torch.save({
            "epoch": epoch,
            "model_state_dict": (ema_model.module if ema_model is not None else model).state_dict(),
            "val_metrics": val_metrics, "model_kwargs": model_kwargs,
            "in_slices": 1, "model_type": args.model_type,
        }, os.path.join(save_dir, "last.pth"))

        # Save checkpoint after EVERY epoch
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_gated_3d": best_gated_3d,
            "history": history,
            "model_kwargs": model_kwargs,
            "model_type": args.model_type,
            "in_slices": 1,
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
        }
        # atomic: a crash mid-write must not leave an unloadable checkpoint behind
        tmp_ckpt = checkpoint_path + ".tmp"
        torch.save(checkpoint_data, tmp_ckpt)
        os.replace(tmp_ckpt, checkpoint_path)
        plot_training_history(history, save_path=history_plot_path)

        # Format clean metric log string
        log_line = (
            f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {train_loss:.4f} | "
            f"3Dall: {val_metrics['dice_3d_gated']:.4f} | 3Dnod: {val_metrics['dice_3d_gated_nod']:.4f} | "
            f"3Draw: {val_metrics['dice_3d_raw_all']:.4f} | CleanOK: {val_metrics['clean_empty_rate']*100:.0f}% | "
            f"gate: {'/'.join(str(x) for x in val_metrics['best_gate']) if val_metrics.get('best_gate') else '-'} | "
            f"2Dall: {val_metrics['dice_all']:.4f} | 2Dtum: {val_metrics['dice']:.4f} | "
            f"FA: {val_metrics['false_alarm_rate']:.4f} | FPcomp: {val_metrics['fp_comp_per_scan']:.2f} | "
            f"Detect: {val_metrics['detect_rate']*100:.1f}% | "
            f"Prec: {val_metrics['precision']:.4f} | Sens: {val_metrics['sensitivity']:.4f} | "
            f"Fail: {val_metrics['failure_rate']*100:.1f}%"
            + ("  <- best" if is_best else "")
        )

        # Print clean single-line metric status to console
        print(log_line, flush=True)

        # Write log entry to results.txt file
        with open(results_log, "a") as f:
            bg = val_metrics.get("best_gate")
            gate_str = f"{bg[0]}/{bg[1]}/{bg[2]}" if bg else "-"
            f.write(f"{epoch:02d} | {train_loss:.4f} | {val_metrics['dice_3d_gated']:.4f} | {val_metrics['dice_3d_gated_nod']:.4f} | {val_metrics['dice_3d_raw_all']:.4f} | {val_metrics['clean_empty_rate']*100:.0f}% | {val_metrics['dice_all']:.4f} | {val_metrics['dice']:.4f} | {val_metrics['false_alarm_rate']:.4f} | {val_metrics['fp_comp_per_scan']:.2f} | {val_metrics['detect_rate']*100:.1f}% | {val_metrics['precision']:.4f} | {val_metrics['sensitivity']:.4f} | {val_metrics['failure_rate']*100:.1f}% | {gate_str}\n")

    plot_training_history(history, save_path=history_plot_path)

if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
