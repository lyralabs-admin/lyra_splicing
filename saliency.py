#!/usr/bin/env python3
# saliency_cli.py

import argparse, importlib.util, os
import numpy as np
import torch
import pysam
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

DNA = np.array(["A","C","G","T"])
BASE_TO_IDX = {"A":0,"C":1,"G":2,"T":3}
_COMP = str.maketrans("ACGTNacgtn","TGCANtgcan")

def one_hot_dna(seq: str) -> np.ndarray:
    x = np.zeros((len(seq), 4), dtype=np.float32)  # [L,4]
    s = seq.upper()
    for i,ch in enumerate(s):
        j = BASE_TO_IDX.get(ch, None)
        if j is not None: x[i, j] = 1.0
    return x

def rc(seq: str) -> str:
    return seq.translate(_COMP)[::-1]  # reverse complement

def fetch_seq_pysam(fasta_path: str, chrom: str, start: int, end: int, strand: str = "+") -> str:
    # start/end 0-based, end-exclusive
    fa = pysam.FastaFile(fasta_path)
    seq = fa.fetch(chrom, int(start), int(end))
    fa.close()
    return rc(seq) if strand == "-" else seq

def load_module(py_path: str):
    spec = importlib.util.spec_from_file_location("lyra_tagger_mod", py_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def get_model_defaults() -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_py = os.path.join(script_dir, "train_lyra_splice_classifier.py")
    m = load_module(model_py)
    return m.MODEL_DEFAULTS

def load_lyra_model(checkpoint_pt: str, device, model_kwargs: dict):
    # fixed model path in same folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_py = os.path.join(script_dir, "train_lyra_splice_classifier.py")
    m = load_module(model_py)
    Model = getattr(m, "LyraSeqTagger")
    model = Model(**model_kwargs).to(device).eval()
    state = torch.load(checkpoint_pt, map_location=device)
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(sd, strict=True)  # let it crash if mismatch
    return model

def parse_positions(s: str) -> list:
    s = s.strip()
    if ":" in s:  # start:end (python-style half-open)
        parts = s.split(":")
        a = int(parts[0]); b = int(parts[1])
        return list(range(a, b))
    return [int(tok) for tok in s.replace(",", " ").split() if tok != ""]

def parse_range(s: str, max_len: int) -> tuple[int, int]:
    parts = s.split(":")
    a = int(parts[0]); b = int(parts[1])
    a = max(0, a); b = min(max_len, b)
    return a, b

def plot_logo_range(mat: np.ndarray, start: int, end: int, out_png: str | None):
    sub = mat[start:end]
    df = pd.DataFrame(sub, columns=["A", "C", "G", "T"])
    ax = logomaker.Logo(df).ax
    ax.set_ylabel("saliency")
    ax.set_xlabel("pos")
    ax.set_title(f"{start}:{end}")
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    else:
        plt.show()

def ism_logo_matrix(seq: str, deltas: np.ndarray) -> np.ndarray:
    neg = -deltas  # higher = more important
    out = np.zeros_like(neg)
    s = seq.upper()
    for i, ch in enumerate(s):
        ref_idx = BASE_TO_IDX.get(ch, None)
        if ref_idx is None:
            continue
        out[i, ref_idx] = float(neg[i].mean())  # avg -delta over all bases
    return out

def _select_scalar(logits: torch.Tensor, target_positions, target_class: int, reduce: str = "sum"):
    # logits: [1,L,3]
    if isinstance(target_positions, slice):
        idx = torch.arange(logits.size(1), device=logits.device)[target_positions]
    else:
        idx = torch.tensor(target_positions, device=logits.device, dtype=torch.long)
    sel = logits[0, idx]  # [K,3]
    if target_class < 0:
        vals = sel.max(dim=-1).values  # argmax per position
    else:
        vals = sel[:, int(target_class)]
    return vals.sum() if reduce == "sum" else vals.mean()

def input_x_grad_matrix(model, seq: str, device, region: tuple[int,int], target_positions, target_class: int):
    x_oh = one_hot_dna(seq)  # [L,4]
    X = torch.from_numpy(x_oh).unsqueeze(0).to(device)  # [1,L,4]
    X.requires_grad_(True)
    logits = model(X)  # [1,L,3]
    score = _select_scalar(logits, target_positions, target_class, reduce="sum")
    model.zero_grad(set_to_none=True)
    score.backward()
    grad = X.grad.detach()[0].cpu().numpy()  # [L,4]
    attr = x_oh * grad  # input*grad per channel
    x0, x1 = region
    return attr[x0:x1].copy()  # [L_window,4] columns A,C,G,T

@torch.no_grad()
def _predict_scalar(model, X: torch.Tensor, target_positions, target_class: int):
    logits = model(X)  # [B,L,3]
    out = []
    for i in range(X.size(0)):
        out.append(_select_scalar(logits[i:i+1], target_positions, target_class, reduce="sum").item())
    return np.array(out, dtype=np.float32)

def ism_matrix(model, seq: str, device, region: tuple[int,int], target_positions, target_class: int, batch_size: int = 256):
    ref_oh = one_hot_dna(seq)
    ref_X = torch.from_numpy(ref_oh).unsqueeze(0).to(device)
    ref_score = float(_predict_scalar(model, ref_X, target_positions, target_class)[0])
    x0, x1 = region
    Lw = x1 - x0
    deltas = np.zeros((Lw, 4), dtype=np.float32)  # [L_window,4]
    s_arr = np.array(list(seq.upper()))
    batch = []
    meta = []
    for i in range(x0, x1):
        for b_idx, b in enumerate(DNA):
            s_mut = s_arr.copy()
            s_mut[i] = b
            batch.append(one_hot_dna("".join(s_mut)))
            meta.append((i - x0, b_idx))
            if len(batch) == batch_size:
                xb = torch.from_numpy(np.stack(batch)).to(device)
                scores = _predict_scalar(model, xb, target_positions, target_class)  # [B]
                for (pw, bj), sc in zip(meta, scores):
                    deltas[pw, bj] = sc - ref_score
                batch, meta = [], []
    if batch:
        xb = torch.from_numpy(np.stack(batch)).to(device)
        scores = _predict_scalar(model, xb, target_positions, target_class)
        for (pw, bj), sc in zip(meta, scores):
            deltas[pw, bj] = sc - ref_score
    return deltas

def main():
    p = argparse.ArgumentParser(description="lyra saliency: input*grad or ism (fixed model from train_lyra_splice_classifier.py)")
    model_defaults = get_model_defaults()
    p.add_argument("--method", choices=["ig", "ism"], required=True)
    # sequence sources
    p.add_argument("--seq", type=str, default="", help="input dna sequence (acgt); if empty, use --fasta/--chrom/--start/--end")
    p.add_argument("--fasta", type=str, default=None)
    p.add_argument("--chrom", type=str, default=None)
    p.add_argument("--start", type=int, default=None, help="0-based start (inclusive) for fasta fetch")
    p.add_argument("--end", type=int, default=None, help="0-based end (exclusive) for fasta fetch")
    p.add_argument("--strand", type=str, default="+", choices=["+","-"])
    p.add_argument("--checkpoint", type=str, required=True, help=".pt checkpoint path")
    # model kwargs (match your trained config)
    p.add_argument("--d_model", type=int, default=model_defaults["d_model"])
    p.add_argument("--d_state", type=int, default=model_defaults["d_state"])
    p.add_argument("--dropout", type=float, default=model_defaults["dropout"])
    p.add_argument("--num_blocks", type=int, default=model_defaults["num_blocks"])
    p.add_argument("--transposed", action="store_true", default=False)
    # target specification
    p.add_argument("--region_start", type=int, default=None, help="start (inclusive) of output slice")
    p.add_argument("--region_end", type=int, default=None, help="end (exclusive) of output slice")
    p.add_argument("--target_positions", type=str, required=True, help="e.g. '2500:2510' or '10,20,30'")
    p.add_argument("--target_class", type=int, default=-1, help="-1=argmax, else 0/1/2")
    # ism batching
    p.add_argument("--batch_size", type=int, default=256)
    # output
    p.add_argument("--out_npy", type=str, required=True, help="where to save the matrix (.npy)")
    p.add_argument("--plot_logo", action="store_true", default=False)
    p.add_argument("--plot_range", type=str, default="", help="start:end in output window coords")
    p.add_argument("--plot_png", type=str, default="")
    p.add_argument("--ism_logo_ref_only", action="store_true", default=False)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_lyra_model(
        checkpoint_pt=args.checkpoint,
        device=device,
        model_kwargs=dict(d_input=4, d_model=args.d_model, d_state=args.d_state,
                          dropout=args.dropout, num_blocks=args.num_blocks, transposed=args.transposed)
    )

    # get sequence
    if args.seq and len(args.seq) > 0:
        seq = args.seq
    else:
        seq = fetch_seq_pysam(args.fasta, args.chrom, args.start, args.end, args.strand)

    # slice region defaults to full length
    x0 = 0 if args.region_start is None else int(args.region_start)
    x1 = len(seq) if args.region_end is None else int(args.region_end)
    region = (x0, x1)

    tgt_pos = parse_positions(args.target_positions)

    if args.method == "ig":
        mat = input_x_grad_matrix(model, seq, device, region, tgt_pos, args.target_class)  # [L_window,4]
    else:
        mat = ism_matrix(model, seq, device, region, tgt_pos, args.target_class, batch_size=args.batch_size)  # [L_window,4]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npy)), exist_ok=True)
    np.save(args.out_npy, mat)
    print(f"saved {args.method} matrix to {args.out_npy} with shape {mat.shape}  # (L_window, 4) A,C,G,T")
    if args.plot_logo:
        pr = args.plot_range if args.plot_range else f"0:{mat.shape[0]}"
        ps, pe = parse_range(pr, mat.shape[0])
        out_png = args.plot_png if args.plot_png else None
        if args.method == "ism" and args.ism_logo_ref_only:
            plot_mat = ism_logo_matrix(seq, mat)
        elif args.method == "ism":
            plot_mat = -mat
        else:
            plot_mat = mat
        plot_logo_range(plot_mat, ps, pe, out_png)

if __name__ == "__main__":
    main()
