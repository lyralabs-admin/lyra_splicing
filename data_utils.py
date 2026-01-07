import csv
import re
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

# -----------------------------
# Constants
# -----------------------------
DNA_TO_IDX = {'A':0, 'C':1, 'G':2, 'T':3}  # n/others -> all zeros
PAD_LABEL = -100  # ignored index for cross entropy
DEFAULT_CLASS_WEIGHTS = [1.0002961158752441, np.sqrt(6755.595703125), np.sqrt(6758.28271484375)]
OPEN_SPLICE_WINDOW = 5000
H5_WINDOW_LENGTH = 15000

# -----------------------------
# DNA and Label Encoders
# -----------------------------
def one_hot_dna(seq: str) -> np.ndarray:
    # returns [L, 4]
    X = np.zeros((len(seq), 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        idx = DNA_TO_IDX.get(ch, -1)
        if idx >= 0:
            X[i, idx] = 1.0
    return X

def parse_label_string(s: str) -> np.ndarray:
    # accepts either contiguous digits like "001020..." or space/comma-separated tokens
    s = s.strip()
    if ' ' in s or ',' in s or '\t' in s:
        tokens = s.replace(',', ' ').split()
        y = np.array(list(map(int, tokens)), dtype=np.int64)
    else:
        y = np.array([int(ch) for ch in s], dtype=np.int64)
    return y  # values in {0,1,2}

# -----------------------------
# Legacy CSV Dataloading
# -----------------------------
def load_csv(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    Xs, Ys = [], []
    with open(path, 'r') as f:
        rows = list(csv.reader(f))[1:]  # skip header
    for r in rows:
        seq = r[0].upper().replace('U', 'T')
        y = parse_label_string(r[1])
        assert len(seq) == len(y)
        Xs.append(one_hot_dna(seq))
        Ys.append(y)
    return Xs, Ys

class SeqDataset(Dataset):
    def __init__(self, Xs: List[np.ndarray], Ys: List[np.ndarray]):
        self.Xs = Xs
        self.Ys = Ys
    def __len__(self):
        return len(self.Xs)
    def __getitem__(self, idx):
        return self.Xs[idx], self.Ys[idx]

def collate_batch(batch):
    # pad to max length in batch
    max_len = max(x.shape[0] for x, _ in batch)
    B = len(batch)
    X_batch = np.zeros((B, max_len, 4), dtype=np.float32)
    y_batch = np.full((B, max_len), PAD_LABEL, dtype=np.int64)
    mask = np.zeros((B, max_len), dtype=np.bool_)
    for i, (x, y) in enumerate(batch):
        L = x.shape[0]
        X_batch[i, :L] = x
        y_batch[i, :L] = y
        mask[i, :L] = True
    return torch.from_numpy(X_batch), torch.from_numpy(y_batch), torch.from_numpy(mask)


# -----------------------------
# H5 Window Dataset
# -----------------------------
class H5WindowDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        center_bp: int = 0,
        input_window_bp: int = 0,
        supervised_window_bp: int = OPEN_SPLICE_WINDOW,
    ):
        self.h5_path = h5_path
        self.center_bp = center_bp
        self.input_window_bp = input_window_bp
        self.supervised_window_bp = supervised_window_bp
        self._hf = None  # lazy-open per worker
        # scan keys and sizes without keeping a shared handle
        with h5py.File(h5_path, 'r') as f:
            self._x_keys = sorted([k for k in f.keys() if re.match(r'^X\d+$', k)], key=lambda s: int(s[1:]))
            self._y_keys = [f"Y{k[1:]}" for k in self._x_keys]
            self._shard_sizes = [f[xk].shape[0] for xk in self._x_keys]
        self._offsets = [0]
        for n in self._shard_sizes:
            self._offsets.append(self._offsets[-1] + n)
        self._length = self._offsets[-1]

    def _ensure_open(self):
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, 'r')

    def __del__(self):
        try:
            if getattr(self, "_hf", None) is not None:
                self._hf.close()
        except Exception:
            pass

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        # find shard
        self._ensure_open()
        # binary search offsets
        lo, hi = 0, len(self._shard_sizes)
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._offsets[mid+1]:
                hi = mid
            else:
                lo = mid + 1
        shard = lo
        local = idx - self._offsets[shard]
        X = self._hf[self._x_keys[shard]][local]       # (SL+CL, 4) int8
        Y = self._hf[self._y_keys[shard]][:]           # (N, SL, 3) or (1, N, SL, 3)
        if Y.ndim == 4 and Y.shape[0] == 1:
            Y = Y[0]
        Y = Y[local]                                  # (SL, 3)
        desired_len = self.input_window_bp if (self.input_window_bp and self.input_window_bp > 0) else X.shape[0]
        desired_len = min(desired_len, H5_WINDOW_LENGTH)
        if X.shape[0] >= desired_len:
            start = (X.shape[0] - desired_len) // 2
            X = X[start:start+desired_len, :]
        else:
            pad_total = desired_len - X.shape[0]
            left = pad_total // 2
            right = pad_total - left
            X = np.pad(X, ((left, right), (0, 0)), mode="constant", constant_values=0)
        input_len = X.shape[0]
        core_len = min(Y.shape[0], OPEN_SPLICE_WINDOW)
        sup_len = min(self.supervised_window_bp, core_len, input_len)
        core_margin = max((core_len - sup_len) // 2, 0)
        Y_sup = Y[core_margin:core_margin+sup_len]
        mask_sup = (Y_sup.sum(axis=-1) > 0).astype(np.bool_)
        y_int = np.full((sup_len,), PAD_LABEL, dtype=np.int64)
        if mask_sup.any():
            y_int[mask_sup] = np.argmax(Y_sup[mask_sup], axis=-1).astype(np.int64)
        y_full = np.full((input_len,), PAD_LABEL, dtype=np.int64)
        mask_full = np.zeros((input_len,), dtype=np.bool_)
        start_full = max((input_len - sup_len) // 2, 0)
        end_full = start_full + sup_len
        y_full[start_full:end_full] = y_int
        mask_full[start_full:end_full] = mask_sup[: end_full - start_full]
        return X.astype(np.float32), y_full, mask_full

def collate_h5(batch):
    # X: (L_in,4), y_int: (L_in,), mask: (L_in,)
    Xs, ys, ms = zip(*batch)
    X = torch.from_numpy(np.stack(Xs, axis=0))         # (B, L_in, 4)
    y = torch.from_numpy(np.stack(ys, axis=0))         # (B, L_in)
    m = torch.from_numpy(np.stack(ms, axis=0))         # (B, L_in)
    return X, y, m

def _shift_tensor(t: torch.Tensor, shift: int, pad_value):
    if shift == 0 or t.size(1) == 0:
        return
    new = t.clone()
    if shift > 0:
        if shift >= t.size(1):
            new.fill_(pad_value)
        else:
            new[:, shift:] = t[:, :-shift]
            new[:, :shift] = pad_value
    else:
        shift = -shift
        if shift >= t.size(1):
            new.fill_(pad_value)
        else:
            new[:, :-shift] = t[:, shift:]
            new[:, -shift:] = pad_value
    t.copy_(new)

def estimate_class_weights(dataset):
    counts = np.zeros(3, dtype=np.float64)
    def iter_samples(ds):
        if isinstance(ds, Subset):
            base = ds.dataset
            for idx in ds.indices:
                yield base[idx]
        else:
            for i in range(len(ds)):
                yield ds[i]
    for sample in iter_samples(dataset):
        if len(sample) >= 3:
            y = np.asarray(sample[1])
            mask = np.asarray(sample[2]).astype(np.bool_)
            y = y[mask]
        else:
            y = np.asarray(sample[1])
        valid = y[(y >= 0) & (y < 3)]
        if valid.size == 0:
            continue
        vals, freq = np.unique(valid, return_counts=True)
        for v, f in zip(vals, freq):
            counts[v] += f
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / counts
    return torch.tensor(weights, dtype=torch.float32)

