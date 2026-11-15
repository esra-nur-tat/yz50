"""Show what this model actually does, in three modes.

The model computes one thing: P(next base | preceding bases). Everything
below reads that same distribution differently.

  1. Unconditional generation -- give it nothing, it writes a sequence.
  2. Prompt continuation -- give it the first half of a real promoter, it
     writes the second half. This is the week 7 and 8 setup, with a new
     alphabet.
  3. Per-position readout -- give it a whole sequence and read the
     distribution at each position without sampling: what did the model
     expect, how sure was it, was it right?

Usage:
    python3 -u sample.py
    python3 -u sample.py --prompt_len 125 --n 4
"""

import argparse
import json
import os

import numpy as np
import torch

from model_dna import DNAConfig, DNAGPT, NATS_TO_BITS

from analyze import load_model, pick_device


def gc(s):
    u = sum(1 for c in s if c in "ACGT")
    return sum(1 for c in s if c in "GC") / u if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--prompt_len", type=int, default=125)
    ap.add_argument("--readout_positions", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    ckpt = args.ckpt or os.path.join(args.results, "best.pt")
    model, ck = load_model(ckpt, device)
    meta = ck["meta"]
    itos = {int(k): v for k, v in meta["itos"].items()}
    L = meta["seq_length"]
    decode = lambda row: "".join(itos[int(i)] for i in row)

    test_pos = np.load(os.path.join(args.data, "test_pos.npy"))
    real_gc = np.mean([gc(decode(r)) for r in test_pos[:500]])

    print(f"device: {device} | checkpoint step {ck['step']}")
    print(f"real promoter GC (500 test sequences): {real_gc:.4f}\n")

    # ---- 1. unconditional ----
    print("=" * 70)
    print("1. UNCONDITIONAL GENERATION -- input: nothing but BOS")
    print("=" * 70)
    gen = model.generate(n=args.n, length=L, device=device).cpu().numpy()
    for i, row in enumerate(gen):
        s = decode(row)
        print(f"\n>generated_{i}  GC={gc(s):.4f}")
        for j in range(0, len(s), 80):
            print(f"  {s[j:j + 80]}")
    print(f"\nmean generated GC = {np.mean([gc(decode(r)) for r in gen]):.4f} "
          f"vs real {real_gc:.4f}")

    # ---- 2. prompt continuation ----
    k = args.prompt_len
    print("\n" + "=" * 70)
    print(f"2. PROMPT CONTINUATION -- input: first {k} bp of a real promoter")
    print("=" * 70)
    idx = np.argsort([0])  # placeholder, replaced below
    picks = test_pos[:args.n]
    prompt = torch.from_numpy(picks[:, :k])
    cont = model.generate(n=args.n, length=L, device=device,
                          prompt=prompt).cpu().numpy()

    for i in range(args.n):
        real = decode(picks[i])
        made = decode(cont[i])
        assert made[:k] == real[:k], "prompt was not preserved"
        print(f"\n--- sequence {i} ---")
        print(f"  prompt      ({k} bp, GC={gc(real[:k]):.3f}): ...{real[max(0,k-60):k]}")
        print(f"  model wrote ({L-k} bp, GC={gc(made[k:]):.3f}): {made[k:k+60]}...")
        print(f"  real was    ({L-k} bp, GC={gc(real[k:]):.3f}): {real[k:k+60]}...")

    made_gc = np.mean([gc(decode(cont[i])[k:]) for i in range(args.n)])
    real_cont_gc = np.mean([gc(decode(picks[i])[k:]) for i in range(args.n)])
    print(f"\ncontinuation GC: model {made_gc:.4f} vs real {real_cont_gc:.4f}")
    print("NOTE: matching GC is not the same as matching structure. The model")
    print("      reproduces composition; whether it reproduces motifs is what")
    print("      the mutagenesis map addresses.")

    # ---- 3. per-position readout, no sampling ----
    print("\n" + "=" * 70)
    print("3. PER-POSITION READOUT -- input: a whole real sequence, no sampling")
    print("=" * 70)
    seq = test_pos[0:1]
    seq_t = torch.from_numpy(seq).to(device)
    x, y = model.make_xy(seq_t)
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits[0].float(), dim=-1).cpu().numpy()  # (L, V)
    actual = y[0].cpu().numpy()

    base_ids = [i for i in sorted(itos) if itos[i] in "ACGT"]
    names = [itos[i] for i in base_ids]
    print(f"\nfirst {args.readout_positions} positions of test sequence 0:")
    print(f"  {'pos':>4} {'actual':>7} {'predicted':>10} {'p(actual)':>10} "
          f"{'entropy':>8}  " + " ".join(f"{n:>6}" for n in names))
    correct = 0
    for p in range(args.readout_positions):
        row = probs[p]
        pr = np.array([row[i] for i in base_ids])
        pr = pr / pr.sum()                       # renormalise over ACGT
        pred = names[int(np.argmax(pr))]
        act = itos[int(actual[p])]
        correct += (pred == act)
        ent = float(-(pr * np.log2(np.clip(pr, 1e-12, 1))).sum())
        print(f"  {p:>4} {act:>7} {pred:>10} {row[int(actual[p])]:>10.4f} "
              f"{ent:>8.3f}  " + " ".join(f"{v:>6.3f}" for v in pr))

    # whole-sequence accuracy and loss, for context
    pr_all = probs[:, base_ids]
    pr_all = pr_all / pr_all.sum(axis=1, keepdims=True)
    pred_all = np.array(base_ids)[np.argmax(pr_all, axis=1)]
    acc = float((pred_all == actual).mean())
    nll = float(-np.log(np.clip(probs[np.arange(len(actual)), actual], 1e-12, 1)).mean())
    print(f"\n  argmax accuracy over all {L} positions: {acc:.4f} "
          f"(a model guessing the commonest base would get "
          f"{max(np.bincount(actual, minlength=max(base_ids)+1)) / L:.4f} here)")
    print(f"  mean NLL: {nll:.4f} nats = {nll * NATS_TO_BITS:.4f} bits/base")
    print("\n  This is the readout AUROC is built on: no sampling, just the")
    print("  probability the model assigns to the bases that are actually there.")

    out = os.path.join(args.results, "sample_report.txt")
    print(f"\n(tip: python3 -u sample.py > {out} to save this)")


if __name__ == "__main__":
    main()
