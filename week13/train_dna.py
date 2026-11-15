"""Generative training on promoter sequences.

Trains only on POSITIVE (promoter) sequences. Negatives are never seen during
training; they are used only by analyze.py for evaluation. That is what makes
the downstream discrimination label-free.

Usage:
    python3 -u train_dna.py --minutes 35
    python3 -u train_dna.py --minutes 35 --n_embd 192 --n_layer 4   # smaller
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from model_dna import (DNAConfig, DNAGPT, NATS_TO_BITS, RANDOM_BITS_PER_BASE,
                       SDPA_OK)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--minutes", type=float, default=35.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=40)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train_on", choices=["pos", "neg"], default="pos",
                    help="which class to model. 'neg' trains the second model "
                         "needed for likelihood-ratio scoring.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device} | fused attention (SDPA): {SDPA_OK}")

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    L = meta["seq_length"]
    vocab_size = meta["vocab_size"]
    print(f"vocab_size={vocab_size}, seq_length={L}, stoi={meta['stoi']}")

    pos = np.load(os.path.join(args.data, f"train_{args.train_on}.npy"))
    print(f"modelling the {args.train_on.upper()} class")
    rng = np.random.default_rng(args.seed)
    pos = pos[rng.permutation(len(pos))]    # the file may be ordered
    n_val = max(1, int(args.val_frac * len(pos)))
    val_np, train_np = pos[:n_val], pos[n_val:]
    print(f"train {len(train_np)} sequences ({len(train_np) * L / 1e6:.2f} Mbase), "
          f"val {len(val_np)} sequences")

    train_t = torch.from_numpy(train_np).to(device)
    val_t = torch.from_numpy(val_np).to(device)

    cfg = DNAConfig(vocab_size=vocab_size, block_size=L, n_embd=args.n_embd,
                    n_head=args.n_head, n_layer=args.n_layer,
                    dropout=args.dropout, bos_id=meta["bos_id"])
    model = DNAGPT(cfg).to(device)
    print(f"parameters: {model.num_params() / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tokens_per_step = args.batch * L
    steps_per_epoch = max(1, round(len(train_np) * L / tokens_per_step))
    print(f"{tokens_per_step:,} tokens/step -> about {steps_per_epoch} steps per epoch")

    def get_batch(split):
        src = train_t if split == "train" else val_t
        ix = torch.randint(len(src), (args.batch,), device=device)
        return src[ix]

    @torch.no_grad()
    def estimate(split):
        model.eval()
        tot = 0.0
        for _ in range(args.eval_iters):
            x, y = model.make_xy(get_batch(split))
            _, loss = model(x, y)
            tot += loss.item()
        model.train()
        return tot / args.eval_iters

    rows = []
    best_val = float("inf")
    tokens = 0
    train_seconds = 0.0       # optimizer steps only; evaluation excluded
    budget = args.minutes * 60.0
    t_wall = time.perf_counter()
    step = 0

    print(f"\nbudget {args.minutes:.0f} min. random baseline "
          f"{RANDOM_BITS_PER_BASE:.2f} bits/base.\n")

    interrupted = False
    try:
        while train_seconds < budget:
            if step % args.eval_every == 0:
                tr_l, va_l = estimate("train"), estimate("val")
                bpb = va_l * NATS_TO_BITS
                rows.append({
                    "step": step, "train_seconds": round(train_seconds, 2),
                    "tokens": tokens, "train_loss": round(tr_l, 5),
                    "val_loss": round(va_l, 5), "val_bits_per_base": round(bpb, 5),
                })
                flag = ""
                if va_l < best_val:
                    best_val = va_l
                    torch.save({"model": model.state_dict(), "config": vars(cfg),
                                "meta": meta, "step": step, "val_loss": va_l},
                               os.path.join(args.out, "best.pt"))
                    flag = "  <- best"
                print(f"step {step:>6} | train {tr_l:.4f} | val {va_l:.4f} | "
                      f"{bpb:.4f} bits/base | {tokens / 1e6:.2f}M tokens | "
                      f"epoch {tokens / (len(train_np) * L):.1f} | "
                      f"{train_seconds / 60:.1f} min{flag}")

            seqs = get_batch("train")
            x, y = model.make_xy(seqs)
            sync(device)
            t0 = time.perf_counter()
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sync(device)
            train_seconds += time.perf_counter() - t0
            tokens += tokens_per_step
            step += 1
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Ctrl-C] training stopped early -- still writing results...")

    # final measurement
    tr_l, va_l = estimate("train"), estimate("val")
    rows.append({"step": step, "train_seconds": round(train_seconds, 2),
                 "tokens": tokens, "train_loss": round(tr_l, 5),
                 "val_loss": round(va_l, 5),
                 "val_bits_per_base": round(va_l * NATS_TO_BITS, 5)})
    if va_l < best_val:
        best_val = va_l
        torch.save({"model": model.state_dict(), "config": vars(cfg),
                    "meta": meta, "step": step, "val_loss": va_l},
                   os.path.join(args.out, "best.pt"))
    torch.save({"model": model.state_dict(), "config": vars(cfg),
                "meta": meta, "step": step, "val_loss": va_l},
               os.path.join(args.out, "last.pt"))

    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- loss curve, in bits per base ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    tr = [r["train_loss"] * NATS_TO_BITS for r in rows]
    va = [r["val_loss"] * NATS_TO_BITS for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, tr, label="train")
    ax.plot(steps, va, label="validation")
    ax.axhline(RANDOM_BITS_PER_BASE, ls="--", c="gray",
               label=f"random ({RANDOM_BITS_PER_BASE:.1f} bits/base)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("bits per base")
    ax.set_title("Promoter-GPT: loss per base")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "loss.png"), dpi=150)

    # --- samples and a GC sanity check ---
    itos = {int(k): v for k, v in meta["itos"].items()}
    model.eval()
    gen = model.generate(n=8, length=L, device=device).cpu().numpy()
    gen_seqs = ["".join(itos[int(i)] for i in row) for row in gen]

    def gc(s):
        u = sum(1 for c in s if c in "ACGT")
        return sum(1 for c in s if c in "GC") / u if u else float("nan")

    real = ["".join(itos[int(i)] for i in row) for row in val_np[:200]]
    gc_gen, gc_real = np.mean([gc(s) for s in gen_seqs]), np.mean([gc(s) for s in real])

    with open(os.path.join(args.out, "samples.txt"), "w") as f:
        f.write(f"generated GC = {gc_gen:.4f}\nreal GC      = {gc_real:.4f}\n\n")
        for i, s in enumerate(gen_seqs):
            f.write(f">generated_{i} GC={gc(s):.4f}\n{s}\n")

    wall = time.perf_counter() - t_wall
    head = "STOPPED EARLY" if interrupted else "done"
    print(f"\n{head}. {step} steps, {tokens / 1e6:.2f}M tokens, "
          f"{tokens / (len(train_np) * L):.1f} epochs")
    print(f"best val {best_val:.4f} nats = {best_val * NATS_TO_BITS:.4f} bits/base")
    print(f"training steps {train_seconds / 60:.1f} min, wall {wall / 60:.1f} min, "
          f"{tokens / train_seconds:,.0f} tokens/s")
    print(f"generated GC {gc_gen:.4f} vs real GC {gc_real:.4f}")
    print(f"-> {args.out}/ : best.pt, last.pt, metrics.csv, loss.png, samples.txt")


if __name__ == "__main__":
    main()
