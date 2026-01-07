#!/usr/bin/env python3
# train lyra per-base splice classifier on openspliceai csv splits

import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split

# This block allows the script to be run directly from the lyra_splicing directory
if __name__ == '__main__' and __package__ is None:
    # To allow running this script directly from the repo root
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from data_utils import H5WindowDataset, collate_h5, _shift_tensor, estimate_class_weights, PAD_LABEL, DEFAULT_CLASS_WEIGHTS, OPEN_SPLICE_WINDOW
from models import LyraSeqTagger
from utils import FocalLoss, evaluate_metrics, classification_metrics, eval_lyra_on_h5

# -----------------------------
# train / eval loops
# -----------------------------
def train_epoch(
    model,
    loader,
    opt,
    device,
    loss_fn,
    jitter_bp=0,
    mask_prob=0.0,
    mask_span=0,
    mask_spans_per_seq=1,
    mask_random_bases=False,
    disable_tqdm: bool = False,
):
    model.train()
    loss_sum = 0.0
    n_tokens = 0
    for X, y, mask in tqdm(loader, desc="train", leave=False, disable=disable_tqdm):
        X = X.to(device, dtype=torch.float32)
        y = y.to(device)
        mask = mask.to(device)
        if jitter_bp > 0:
            max_shift = jitter_bp
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=X.device).item())
            if shift != 0:
                _shift_tensor(X, shift, 0.0)
                _shift_tensor(y, shift, PAD_LABEL)
                _shift_tensor(mask, shift, False)
        if mask_prob > 0.0 and mask_span > 0:
            span = min(mask_span, X.size(1))
            if span > 0:
                rand_mask = torch.rand(X.size(0), device=X.device) < mask_prob
                if rand_mask.any():
                    idxs = torch.nonzero(rand_mask, as_tuple=False).squeeze(-1).tolist()
                    for i in idxs:
                        for _ in range(max(1, mask_spans_per_seq)):
                            start_max = max(0, X.size(1) - span)
                            if start_max == 0:
                                start = 0
                            else:
                                start = torch.randint(0, start_max + 1, (1,), device=X.device).item()
                            if mask_random_bases:
                                rand_bases = torch.randint(0, X.size(2), (span,), device=X.device)
                                X[i, start:start+span, :] = 0.0
                                X[i, start:start+span, :].scatter_(1, rand_bases.unsqueeze(-1), 1.0)
                            else:
                                X[i, start:start+span, :] = 0.0
        logits = model(X)  # [B, L_in, 3]
        L_in = logits.size(1)
        if L_in > y.size(1):
            off = (L_in - y.size(1)) // 2
            logits = logits[:, off:off+y.size(1), :]
        loss = loss_fn(logits, y, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tokens = int(mask.sum().item())
        loss_sum += loss.item() * tokens
        n_tokens += tokens
    return loss_sum / max(1, n_tokens)

@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    center_bp: int = 0,
    loss_fn=None,
    disable_tqdm: bool = False,
):
    model.eval()
    loss_sum, n_tokens = 0.0, 0
    all_logits, all_labels, all_masks = [], [], []
    for X, y, mask in tqdm(loader, desc="val", leave=False, disable=disable_tqdm):
        X = X.to(device, dtype=torch.float32)
        y = y.to(device)
        logits = model(X)  # (B, L_in, 3)
        L_in = logits.size(1)
        L_tgt = y.size(1)
        if L_in < L_tgt:
            off_y = (L_tgt - L_in) // 2
            y_aligned = y[:, off_y:off_y+L_in]
            mask_aligned = mask[:, off_y:off_y+L_in]
            logits_aligned = logits
        else:
            off_l = (L_in - L_tgt) // 2
            logits_aligned = logits[:, off_l:off_l+L_tgt, :]
            y_aligned = y
            mask_aligned = mask
        if logits_aligned.size(1) > OPEN_SPLICE_WINDOW:
            off_w = (logits_aligned.size(1) - OPEN_SPLICE_WINDOW) // 2
            logits_aligned = logits_aligned[:, off_w:off_w+OPEN_SPLICE_WINDOW, :]
            y_aligned = y_aligned[:, off_w:off_w+OPEN_SPLICE_WINDOW]
            mask_aligned = mask_aligned[:, off_w:off_w+OPEN_SPLICE_WINDOW]
        # optional further center crop for evaluation metrics
        if center_bp and center_bp > 0 and logits_aligned.size(1) >= center_bp:
            o2 = (logits_aligned.size(1) - center_bp) // 2
            logits_eval = logits_aligned[:, o2:o2+center_bp, :]
            y_eval = y_aligned[:, o2:o2+center_bp]
            mask_eval = mask_aligned[:, o2:o2+center_bp]
        else:
            logits_eval, y_eval, mask_eval = logits_aligned, y_aligned, mask_aligned
        if loss_fn is not None:
            loss = loss_fn(logits_aligned, y_aligned, mask_aligned)
        else:
            loss = torch.tensor(0.0, device=device)
        tokens = int(mask_aligned.sum().item())
        loss_sum += loss.item() * tokens
        n_tokens += tokens
        all_logits.append(logits_eval.cpu())
        all_labels.append(y_eval.cpu())
        all_masks.append(mask_eval.cpu())
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    masks_cat = torch.cat(all_masks, dim=0)
    topk = evaluate_metrics(logits_cat, labels_cat, masks_cat)
    cls_metrics = classification_metrics(logits_cat, labels_cat, masks_cat)
    merged = {**topk, **cls_metrics}
    return loss_sum / max(1, n_tokens), merged

# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Lyra Splice Classifier")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=48, help="Model dimension")
    parser.add_argument("--d_state", type=int, default=48, help="SSM state dimension")
    parser.add_argument("--num_blocks", type=int, default=22, help="Number of Lyra blocks")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")

    # Dataset and training parameters
    parser.add_argument("--train_h5", type=str, default="train_test_dataset_MANE/dataset_train.h5", help="Path to training H5 file")
    parser.add_argument("--test_h5", type=str, default="train_test_dataset_MANE/dataset_test.h5", help="Path to test H5 file")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--out_dir", type=str, default="weights", help="Output directory for checkpoints")
    parser.add_argument("--save_best_metric", type=str, default="avg_topk", choices=["avg_topk", "val_loss"], help="Metric to monitor for saving best model")
    parser.add_argument("--center_bp", type=int, default=5000, help="Center window size for evaluation")
    parser.add_argument("--supervised_window_bp", type=int, default=5000, help="Size of labeled center region for loss")
    parser.add_argument("--input_window_bp", type=int, default=0, help="Center crop input X to this size (overrides flank)")
    parser.add_argument("--input_flank_bp", type=int, default=5000, help="Extra flank bp around the core window")
    parser.add_argument("--train_jitter_bp", type=int, default=5, help="Max random jitter for training crops")
    
    # Optimizer parameters
    parser.add_argument("--lr_init", type=float, default=1e-3, help="Initial learning rate for AdamW")
    parser.add_argument("--lr_final", type=float, default=1e-5, help="Final learning rate for cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")

    # DDP and early stopping
    parser.add_argument("--ddp", action="store_true", default=False, help="Enable Distributed Data Parallel training")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs")
    
    # Loss and class weights
    parser.add_argument("--disable_class_weights", action="store_true", default=False, help="Disable class weighting in the loss")
    parser.add_argument("--class_weight_cache", type=str, default=None, help="Path to cached class weights (.npy)")
    parser.add_argument("--recompute_class_weights", action="store_true", default=False, help="Recompute and cache class weights")
    parser.add_argument("--class_weights", type=float, nargs=3, default=None, help="Manually override class weights")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["ce", "focal"], help="Loss function type")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss")
    parser.add_argument("--label_smoothing", type=float, default=0.00, help="Label smoothing factor")

    # Augmentations
    parser.add_argument("--mask_prob", type=float, default=0.15, help="Probability of applying span masking")
    parser.add_argument("--mask_span", type=int, default=5, help="Length of masked spans in bp")
    parser.add_argument("--mask_spans_per_seq", type=int, default=5, help="Number of spans to mask per sequence")
    parser.add_argument("--mask_random_bases", action="store_true", default=False, help="Use random bases for masking instead of zeros")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # DDP init
    ddp_env = ("RANK" in os.environ or "LOCAL_RANK" in os.environ)
    ddp = args.ddp or ddp_env
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if ddp else 0
    if ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = (not ddp) or (dist.get_rank() == 0)

    # load H5 dataset and split 90/10 for validation
    if args.input_window_bp and args.input_window_bp > 0:
        effective_input_bp = args.input_window_bp
    else:
        flank = max(0, args.input_flank_bp)
        effective_input_bp = min(OPEN_SPLICE_WINDOW + flank, 15000)  # dataset length safeguard
    full_ds = H5WindowDataset(
        args.train_h5,
        center_bp=args.center_bp,
        input_window_bp=effective_input_bp,
        supervised_window_bp=args.supervised_window_bp,
    )
    val_size = max(1, int(0.10 * len(full_ds)))
    train_size = len(full_ds) - val_size
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=g)

    class_weights = None
    if not args.disable_class_weights:
        weights_np = None
        if args.class_weights is not None:
            weights_np = np.array(args.class_weights, dtype=np.float32)
            if is_main:
                print("using class weights from --class_weights")
        elif args.recompute_class_weights:
            if is_main:
                print("recomputing class weights from train split...")
                weights_tensor = estimate_class_weights(train_ds)
                weights_np = weights_tensor.numpy().astype(np.float32, copy=False)
                if args.class_weight_cache:
                    np.save(args.class_weight_cache, weights_np)
                    print(f"saved class weights to {args.class_weight_cache}")
        elif args.class_weight_cache and os.path.exists(args.class_weight_cache):
            weights_np = np.load(args.class_weight_cache).astype(np.float32, copy=False)
            if is_main:
                print(f"loaded class weights from {args.class_weight_cache}")
        else:
            weights_np = np.array(DEFAULT_CLASS_WEIGHTS, dtype=np.float32)
            if is_main:
                print("using default class weights")

        if ddp:
            class_weights = torch.zeros(3, dtype=torch.float32, device=device)
            if weights_np is not None:
                class_weights.copy_(torch.from_numpy(weights_np).to(device))
            dist.broadcast(class_weights, src=0)
        else:
            if weights_np is None:
                weights_np = np.array(DEFAULT_CLASS_WEIGHTS, dtype=np.float32)
            class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)

    # Samplers
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
        shuffle_flag_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_flag_train = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle_flag_train,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=1,
        collate_fn=collate_h5,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=1,
        collate_fn=collate_h5,
    )

    if is_main:
        print(f"device: {device}")
        if class_weights is not None:
            print(f"class weights: {class_weights.tolist()}")

    if class_weights is not None:
        class_weights = class_weights.to(device)

    smoothing = max(0.0, min(args.label_smoothing, 1.0))
    num_classes = 3
    if args.loss_type == "focal":
        base_criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights, ignore_index=PAD_LABEL).to(device)
    else:
        base_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=PAD_LABEL).to(device)

    def loss_fn(logits, targets, mask):
        logits_flat = logits[mask]
        targets_flat = targets[mask]
        if logits_flat.numel() == 0:
            return logits.sum() * 0
        if smoothing <= 0.0:
            return base_criterion(logits_flat, targets_flat)
        with torch.no_grad():
            true_dist = torch.full((logits_flat.size(0), num_classes), smoothing / (num_classes - 1), device=logits_flat.device, dtype=logits_flat.dtype)
            true_dist.scatter_(1, targets_flat.unsqueeze(1), 1.0 - smoothing)
        log_probs = torch.log_softmax(logits_flat, dim=-1)
        if class_weights is not None:
            weights = class_weights.to(logits_flat.device)
            loss = -(true_dist * log_probs * weights.unsqueeze(0)).sum(dim=-1)
        else:
            loss = -(true_dist * log_probs).sum(dim=-1)
        return loss.mean()

    # model
    model = LyraSeqTagger(
        d_input=4,
        d_model=args.d_model,
        d_state=args.d_state,
        dropout=args.dropout,
        num_blocks=args.num_blocks,
        transposed=False
    ).to(device)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        raw_model = model.module
    else:
        raw_model = model
        
    num_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        print(f"trainable parameters: {num_params:,}")
        
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, args.epochs),
        eta_min=args.lr_final,
    )

    epochs = args.epochs
    best_val = float("-inf") if args.save_best_metric == "avg_topk" else float("inf")
    best_state = None
    best_info = {}
    no_improve = 0

    for ep in range(1, epochs+1):
        if ddp and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(ep)
        t0 = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            opt,
            device,
            loss_fn,
            jitter_bp=args.train_jitter_bp,
            mask_prob=args.mask_prob,
            mask_span=args.mask_span,
            mask_spans_per_seq=args.mask_spans_per_seq,
            mask_random_bases=args.mask_random_bases,
            disable_tqdm=not is_main
        )
        val_loss, val_metrics = eval_epoch(
            model,
            val_loader,
            device,
            center_bp=args.center_bp,
            loss_fn=loss_fn,
            disable_tqdm=not is_main,
        )
        current_lr = opt.param_groups[0]["lr"]
        if is_main:
            print(
                f"epoch {ep} | train ce/token {train_loss:.6f} | val ce/token {val_loss:.6f} | "
                f"acc_topk {val_metrics['acceptor topk acc']:.4f} | don_topk {val_metrics['donor topk acc']:.4f} | "
                f"avg_topk {val_metrics['avg topk acc']:.4f} | "
                f"acc_auprc {val_metrics.get('acceptor auprc',0.0):.4f} | don_auprc {val_metrics.get('donor auprc',0.0):.4f} | "
                f"avg_auprc {val_metrics.get('avg auprc',0.0):.4f} | "
                f"acc_f1 {val_metrics.get('acceptor_f1',0.0):.4f} | don_f1 {val_metrics.get('donor_f1',0.0):.4f} | "
                f"macro_f1 {val_metrics.get('macro_f1_acceptor_donor',0.0):.4f} | "
                f"token_acc {val_metrics.get('token_accuracy',0.0):.4f} | "
                f"time {time.time()-t0:.1f}s | lr {current_lr:.2e}"
            )
        current_score = val_metrics['avg topk acc'] if args.save_best_metric == "avg_topk" else -val_loss
        improved = current_score > best_val
        if improved and is_main:
            best_val = current_score
            best_state = {k: v.cpu() for k, v in raw_model.state_dict().items()}
            best_info = {"epoch": ep, "val_loss": val_loss, **val_metrics}
            torch.save(
                {"state_dict": best_state, "info": best_info},
                os.path.join(args.out_dir, "best.pt")
            )
        # update patience
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        # early stopping check synced across ranks
        stop_tensor = torch.tensor(1 if (no_improve >= args.patience and is_main) else 0, device=device)
        if ddp:
            dist.broadcast(stop_tensor, src=0)
        stop_now = stop_tensor.item() == 1
        if stop_now:
            if is_main:
                print(f"early stopping triggered at epoch {ep} (no improvement for {args.patience} epochs)")
            break
        if scheduler is not None:
            scheduler.step()

    if best_state is not None and is_main:
        raw_model.load_state_dict(best_state)

    if is_main:
        # results are stored under args.out_dir: best.pt (best validation by avg_topk or val_loss) and last.pt (final test)
        test_metrics = eval_lyra_on_h5(raw_model, args.test_h5, device, batch_size=args.batch_size, center_bp=args.center_bp)
        print("\nfinal test (H5):")
        print(f"acceptor topk acc: {test_metrics['acceptor topk acc']:.4f}")
        print(f"donor topk acc:    {test_metrics['donor topk acc']:.4f}")
        print(f"avg topk acc:      {test_metrics['avg topk acc']:.4f}")
        torch.save(
            {"state_dict": {k: v.cpu() for k, v in raw_model.state_dict().items()},
             "info": {**test_metrics}},
            os.path.join(args.out_dir, "last.pt")
        )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
