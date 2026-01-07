import heapq
import re
from typing import List, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from data_utils import PAD_LABEL, OPEN_SPLICE_WINDOW

# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def top_k_accuracy_multidimensional(scores: np.ndarray, true_labels: np.ndarray, class_index: int):
    # scores: [num_samples, seq_length, num_classes]
    # true_labels: [num_samples, seq_length]
    class_scores = scores[:, :, class_index].reshape(-1)
    class_true = (true_labels.reshape(-1) == class_index).astype(np.int32)
    k = int(class_true.sum())
    if k == 0:
        return 0.0 # Return 0 if no positive instances, was previously a ValueError
    top_idx = np.argsort(class_scores)[::-1][:k]
    tp = int(class_true[top_idx].sum())
    return tp / k

def evaluate_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    # logits: [B, L, 3], labels: [B, L], mask: [B, L] True for valid
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    y = labels.cpu().numpy()
    m = mask.cpu().numpy()
    # filter masked positions
    probs_f = probs[m]
    y_f = y[m]
    probs_f = probs_f.reshape(1, -1, 3)  # treat as single long sequence
    y_f = y_f.reshape(1, -1)
    acc = []
    auprcs = []
    for ci in [1, 2]:
        try:
            acc.append(top_k_accuracy_multidimensional(probs_f, y_f, ci))
        except ValueError:
            acc.append(0.0) # Handle case with no positive instances
            
        ci_scores = probs[:, :, ci][m].reshape(-1)
        ci_true = (y.reshape(-1) == ci)[m.reshape(-1)]
        if ci_true.any():
            auprcs.append(average_precision_score(ci_true.astype(int), ci_scores))
        else:
            auprcs.append(0.0)
    return {
        "acceptor topk acc": acc[0],
        "donor topk acc": acc[1],
        "avg topk acc": float(np.mean(acc)),
        "acceptor auprc": auprcs[0],
        "donor auprc": auprcs[1],
        "avg auprc": float(np.mean(auprcs)),
    }

def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    # per-position argmax metrics over masked tokens
    with torch.no_grad():
        y_true = labels[mask].cpu().numpy()
        y_pred = torch.argmax(logits[mask], dim=-1).cpu().numpy()
    metrics = {}
    f1s = []
    for cls, name in [(1, "acceptor"), (2, "donor")]:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f"{name}_precision"] = precision
        metrics[f"{name}_recall"] = recall
        metrics[f"{name}_f1"] = f1
        f1s.append(f1)
    metrics["macro_f1_acceptor_donor"] = float(np.mean(f1s))
    # overall accuracy over masked tokens
    metrics["token_accuracy"] = float((y_pred == y_true).mean()) if y_true.size > 0 else 0.0
    return metrics

# -----------------------------
# Loss Function
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, ignore_index: int = PAD_LABEL):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer("weight", weight.clone().detach())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.numel() == 0:
            return logits.sum()
        
        targets = targets.reshape(-1)
        logits = logits.reshape(-1, logits.size(-1))
        
        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return logits.sum() * 0
        
        logits = logits[valid]
        targets = targets[valid]
        
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        if self.weight is not None:
            alpha = self.weight[targets]
        else:
            alpha = 1.0
        
        loss = -alpha * (1 - pt) ** self.gamma * log_pt
        return loss.mean()

# -----------------------------
# Standalone Evaluation
# -----------------------------
@torch.no_grad()
def eval_lyra_on_h5(model: nn.Module, h5_path: str, device, batch_size: int = 8, center_bp: int = 100):
    model.eval()
    # first pass: count positives
    k_acc = 0
    k_don = 0
    with h5py.File(h5_path, 'r') as f:
        ykeys = sorted([k for k in f.keys() if re.match(r'^Y\d+$', k)], key=lambda s: int(s[1:]))
        for yk in ykeys:
            Y = f[yk][:]  # (N, SL, 3) or (1,N,SL,3)
            if Y.ndim == 4 and Y.shape[0] == 1:
                Y = Y[0]
            Yt = torch.from_numpy(Y).float()  # (N, SL, 3)
            if center_bp and center_bp > 0 and Yt.shape[1] >= center_bp:
                o = (Yt.shape[1] - center_bp) // 2
                Yt = Yt[:, o:o+center_bp, :]
            mask = (Yt.sum(dim=-1) > 0)
            y_int = torch.argmax(Yt, dim=-1)
            k_acc += int(((y_int == 1) & mask).sum().item())
            k_don += int(((y_int == 2) & mask).sum().item())
    
    if k_acc == 0 and k_don == 0:
        return {"acceptor topk acc": 0.0, "donor topk acc": 0.0, "avg topk acc": 0.0}
    
    acc_heap: List[Tuple[float, bool]] = []
    don_heap: List[Tuple[float, bool]] = []
    
    def push_heap(heap, k_limit, scores, trues):
        if k_limit == 0: return
        for s, t in zip(scores, trues):
            fs = float(s)
            if len(heap) < k_limit:
                heapq.heappush(heap, (fs, bool(t)))
            elif fs > heap[0][0]:
                heapq.heapreplace(heap, (fs, bool(t)))

    # second pass: predict streaming and update heaps
    with h5py.File(h5_path, 'r') as f:
        xkeys = sorted([k for k in f.keys() if re.match(r'^X\d+$', k)], key=lambda s: int(s[1:]))
        for i, xk in tqdm(enumerate(xkeys), total=len(xkeys), desc="Eval H5", leave=False, ncols=80):
            X = f[xk][:]  # (N, L_in, 4)
            Y = f[f"Y{xk[1:]}"][:]
            if Y.ndim == 4 and Y.shape[0] == 1:
                Y = Y[0]
            N = X.shape[0]
            for s in range(0, N, batch_size):
                e = min(s + batch_size, N)
                xb = torch.tensor(X[s:e], dtype=torch.float32, device=device)  # (b, L_in, 4)
                logits = model(xb).cpu()  # (b, L_in, 3)
                L_in = logits.size(1)
                SL = Y.shape[1]
                off = max((L_in - SL) // 2, 0)
                logits = logits[:, off:off+SL, :]  # center SL
                Yb = torch.from_numpy(Y[s:e])
                if logits.size(1) > OPEN_SPLICE_WINDOW:
                    owin = (logits.size(1) - OPEN_SPLICE_WINDOW) // 2
                    logits = logits[:, owin:owin+OPEN_SPLICE_WINDOW, :]
                    Yb = Yb[:, owin:owin+OPEN_SPLICE_WINDOW, :]
                if center_bp and center_bp > 0 and logits.size(1) >= center_bp:
                    o = (logits.size(1) - center_bp) // 2
                    logits = logits[:, o:o+center_bp, :]
                    Yb = Yb[:, o:o+center_bp, :]
                probs = torch.softmax(logits, dim=-1).numpy()
                y_int = torch.argmax(Yb, dim=-1).numpy()
                mask = (Yb.sum(dim=-1) > 0).numpy()
                valid = mask.reshape(-1)
                if not valid.any(): continue
                p = probs.reshape(-1, 3)[valid]
                yf = y_int.reshape(-1)[valid]
                push_heap(acc_heap, k_acc, p[:, 1], (yf == 1))
                push_heap(don_heap, k_don, p[:, 2], (yf == 2))

    acc_tp = sum(1 for _, t in acc_heap if t)
    don_tp = sum(1 for _, t in don_heap if t)
    acc_topk = (acc_tp / k_acc) if k_acc > 0 else 0.0
    don_topk = (don_tp / k_don) if k_don > 0 else 0.0
    
    return {"acceptor topk acc": acc_topk, "donor topk acc": don_topk, "avg topk acc": float(np.mean([acc_topk, don_topk]))}

