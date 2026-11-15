"""Why the model loses, and what fixes it.

The per-sequence mean loss scores 0.726 AUROC while GC fraction alone scores
0.796. This script tests the four candidate explanations by building the
corresponding scores and comparing them on one held-out split.

Methods are grouped by how much supervision they use, because putting them in
one ranked table would otherwise be misleading:

  Tier 1, no labels anywhere
    - mean loss under the promoter model (the current result)
    - GC fraction (the baseline that beats it)

  Tier 2, labels used to build the model, not to score
    - likelihood ratio: NLL under a negative-class model minus NLL under the
      promoter model. Needs train_dna.py --train_on neg.

  Tier 3, supervised probes on frozen features
    - logistic regression on the 251-dim per-position loss profile
    - logistic regression on the mean-pooled embedding
    - logistic regression on GC and mean loss together

Every number is computed on the same held-out half of the test set.

Usage:
    python3 -u improve.py
"""

import argparse
import json
import os

import numpy as np
import torch

from model_dna import NATS_TO_BITS
from analyze import (auroc, load_model, pick_device, gc_fraction_ids,
                     batched_seq_nll, batched_embed)


@torch.no_grad()
def per_position_profiles(model, arr, device, bs=128):
    out = []
    for i in range(0, len(arr), bs):
        chunk = torch.from_numpy(arr[i:i + bs]).to(device)
        out.append(model.per_position_nll(chunk).float().cpu().numpy())
    return np.concatenate(out)


# --- logistic regression in numpy, so Tier 3 needs no extra install ---

def _standardise(Xf, Xe):
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (Xf - mu) / sd, (Xe - mu) / sd


def _logistic_fit(X, y, C=1.0, iters=3000, lr=0.5):
    """L2-regularised logistic regression by gradient descent with momentum.

    Features are standardised by the caller, so a fixed step size is safe.
    C follows the scikit-learn convention: larger C, weaker regularisation.
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    v = np.zeros(d + 1)
    lam = 1.0 / (C * n)
    for t in range(iters):
        z = np.clip(Xb @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (p - y) / n
        g[:d] += lam * w[:d]                  # no penalty on the bias
        step = lr * (1.0 - t / iters) ** 0.5  # gentle decay
        v = 0.9 * v + g
        w -= step * v
    return w


def _logistic_score(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def fit_probe(Xf, yf, Xe, ye, name, results):
    """Logistic probe. The regularisation strength is chosen on a split of the
    fitting half only; the evaluation half is never touched."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xf))
    Xf, yf = Xf[idx], yf[idx]
    cut = int(0.75 * len(Xf))

    best_C, best_a = 1.0, -1.0
    for C in (0.01, 0.1, 1.0, 10.0):
        Xtr, Xva = _standardise(Xf[:cut], Xf[cut:])
        w = _logistic_fit(Xtr, yf[:cut], C=C)
        a = auroc(_logistic_score(Xva, w), yf[cut:])
        if a > best_a:
            best_C, best_a = C, a

    Xtr, Xte = _standardise(Xf, Xe)
    w = _logistic_fit(Xtr, yf, C=best_C)
    a = auroc(_logistic_score(Xte, w), ye)
    print(f"  {name:<44} AUROC {a:.4f}   (C={best_C}, {Xf.shape[1]} features)")
    results[name] = {"auroc": float(a), "C": best_C,
                     "n_features": int(Xf.shape[1])}
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--neg_results", default="results_neg")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    model, ck = load_model(os.path.join(args.results, "best.pt"), device)
    meta = ck["meta"]
    stoi = meta["stoi"]
    L = meta["seq_length"]
    print(f"device: {device} | promoter model step {ck['step']}\n")

    test_pos = np.load(os.path.join(args.data, "test_pos.npy"))
    test_neg = np.load(os.path.join(args.data, "test_neg.npy"))
    X_all = np.concatenate([test_pos, test_neg])
    y_all = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

    # One split, used by everything, so the numbers are comparable.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]
    half = len(X_all) // 2
    fit_i, ev_i = np.arange(half), np.arange(half, len(X_all))
    print(f"test set {len(X_all)} sequences -> fit {len(fit_i)}, "
          f"evaluate {len(ev_i)}")
    print("Tier 1 and 2 need no fitting; they are still scored on the")
    print("evaluation half alone so every row is comparable.\n")

    print("computing features...")
    nll_mean = batched_seq_nll(model, X_all, device, args.batch)
    profiles = per_position_profiles(model, X_all, device, args.batch)
    embeds = batched_embed(model, X_all, device, args.batch)
    gc = gc_fraction_ids(X_all, stoi)
    print(f"  loss profile {profiles.shape}, embedding {embeds.shape}")

    results = {}
    ye = y_all[ev_i]

    print("\n=== TIER 1: no labels used at any point ===")
    a_loss = auroc(-nll_mean[ev_i], ye)
    a_gc = auroc(gc[ev_i], ye)
    a_gc = max(a_gc, 1.0 - a_gc)
    print(f"  {'mean loss under the promoter model':<44} AUROC {a_loss:.4f}")
    print(f"  {'GC fraction':<44} AUROC {a_gc:.4f}")
    results["tier1_mean_loss"] = {"auroc": float(a_loss)}
    results["tier1_gc"] = {"auroc": float(a_gc)}

    print("\n=== TIER 2: labels used to build a second model, not to score ===")
    negp = os.path.join(args.neg_results, "best.pt")
    if os.path.exists(negp):
        neg_model, neg_ck = load_model(negp, device)
        print(f"  negative-class model: step {neg_ck['step']}, "
              f"val {neg_ck['val_loss'] * NATS_TO_BITS:.4f} bits/base")
        nll_neg_model = batched_seq_nll(neg_model, X_all, device, args.batch)
        lr_score = nll_neg_model - nll_mean          # high = promoter-like
        a_lr = auroc(lr_score[ev_i], ye)
        print(f"  {'likelihood ratio (neg model - pos model)':<44} AUROC {a_lr:.4f}")
        results["tier2_likelihood_ratio"] = {"auroc": float(a_lr)}

        # Does the ratio remove the arch? Correlate each score with GC.
        r_single = float(np.corrcoef(nll_mean, gc)[0, 1])
        r_ratio = float(np.corrcoef(lr_score, gc)[0, 1])
        g, v = gc, lr_score
        quad = np.vstack([g ** 2, g, np.ones(len(g))]).T
        cq = np.linalg.lstsq(quad, v, rcond=None)[0]
        print(f"    correlation with GC: single model {r_single:+.3f}, "
              f"ratio {r_ratio:+.3f}")
        print(f"    curvature of the ratio against GC: {cq[0]:+.3f} "
              f"(the single model's was negative and large)")
        results["tier2_ratio_gc_corr"] = r_ratio
        results["tier2_ratio_gc_curvature"] = float(cq[0])
        np.savez(os.path.join(args.results, "lr_scores.npz"),
                 lr_score=lr_score, gc=gc, y=y_all, nll_pos_model=nll_mean,
                 nll_neg_model=nll_neg_model)
    else:
        print(f"  skipped: no negative-class model at {negp}")
        print("  run: python3 -u train_dna.py --train_on neg "
              "--out results_neg --minutes 20 --eval_iters 20")

    print("\n=== TIER 3: supervised probes on frozen features ===")
    if True:
        two = np.vstack([gc, nll_mean]).T
        fit_probe(two[fit_i], y_all[fit_i], two[ev_i], ye,
                  "GC and mean loss together", results)
        fit_probe(profiles[fit_i], y_all[fit_i], profiles[ev_i], ye,
                  "per-position loss profile", results)
        fit_probe(embeds[fit_i], y_all[fit_i], embeds[ev_i], ye,
                  "mean-pooled embedding", results)
        gconly = gc[:, None]
        fit_probe(gconly[fit_i], y_all[fit_i], gconly[ev_i], ye,
                  "GC alone (probe, for reference)", results)

    out = os.path.join(args.results, "improve_report.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 66)
    print("SUMMARY  (all on the same evaluation half)")
    print("=" * 66)
    for k, v in sorted(results.items(),
                       key=lambda kv: -(kv[1]["auroc"]
                                        if isinstance(kv[1], dict) else 0)):
        if isinstance(v, dict) and "auroc" in v:
            print(f"  {v['auroc']:.4f}  {k}")
    print(f"\nwrote {out}")
    print("\nReminder: tier 3 rows use labels to fit, tier 1 rows do not.")
    print("They answer different questions and should not be ranked together.")


if __name__ == "__main__":
    main()
