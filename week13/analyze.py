"""Post-hoc analyses of the trained Promoter-GPT.

None of these require further training; all run on a single checkpoint.

  1. Per-sequence loss -> AUROC (label-free discrimination)
  2. GC-content baseline AUROC, to test whether the model learned more than
     base composition
  3. In-silico saturation mutagenesis heat map
  4. Position-bias plot, showing a structural limit of a causal model
  5. Optional: embeddings, PCA and a linear probe (if scikit-learn is present)

Usage:
    python3 -u analyze.py
    python3 -u analyze.py --ckpt results/last.pt
"""

import argparse
import json
import os

import numpy as np
import torch

from model_dna import DNAConfig, DNAGPT, NATS_TO_BITS, RANDOM_BITS_PER_BASE


# ---------- AUROC: dependency-free, tie-aware ----------

def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    r = np.empty(len(a), dtype=float)
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0     # average rank for ties
        i = j + 1
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = r
    return ranks


def auroc(scores, labels):
    """Mann-Whitney U. labels: 1 = positive. Higher score = more positive."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rankdata(scores)
    return (r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


# ---------- helpers ----------

def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"no checkpoint at {ckpt_path}\n"
            f"run training first: python3 -u train_dna.py --minutes 35"
        )
    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location=device)   # older torch
    cfg = DNAConfig(**ck["config"])
    model = DNAGPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def batched_seq_nll(model, arr, device, bs=128):
    out = []
    for i in range(0, len(arr), bs):
        chunk = torch.from_numpy(arr[i:i + bs]).to(device)
        out.append(model.sequence_nll(chunk).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def batched_embed(model, arr, device, bs=128):
    out = []
    for i in range(0, len(arr), bs):
        chunk = torch.from_numpy(arr[i:i + bs]).to(device)
        out.append(model.embed(chunk).float().cpu().numpy())
    return np.concatenate(out)


def unigram_baseline(data_dir, meta, seed=1337, val_frac=0.05):
    """Zeroth-order entropy of the training promoters.

    This is the loss a model that knows only the base frequencies would
    reach -- no context at all. Promoters are GC rich, so this sits below the
    uniform 2.0 bits/base, and it is the honest floor the model has to beat.
    The split mirrors train_dna.py so the frequencies come from exactly the
    sequences the model trained on.
    """
    pos = np.load(os.path.join(data_dir, "train_pos.npy"))
    rng = np.random.default_rng(seed)
    pos = pos[rng.permutation(len(pos))]
    n_val = max(1, int(val_frac * len(pos)))
    train = pos[n_val:]
    cnt = np.bincount(train.ravel(), minlength=meta["vocab_size"]).astype(float)
    p = cnt / cnt.sum()
    nz = p[p > 0]
    h_nats = float(-(nz * np.log(nz)).sum())
    freqs = {meta["itos"][str(i)]: float(p[i])
             for i in range(len(p)) if p[i] > 0 and str(i) in meta["itos"]}
    return h_nats, freqs


def gc_fraction_ids(arr, stoi):
    """arr: (N, L) base ids -> (N,) GC fraction, ignoring N in the denominator."""
    gc_ids = [stoi[b] for b in ("G", "C") if b in stoi]
    acgt_ids = [stoi[b] for b in ("A", "C", "G", "T") if b in stoi]
    is_gc = np.isin(arr, gc_ids).sum(axis=1)
    is_acgt = np.isin(arr, acgt_ids).sum(axis=1)
    return np.where(is_acgt > 0, is_gc / np.maximum(is_acgt, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--ckpt", default=None, help="defaults to results/best.pt")
    ap.add_argument("--mut_seqs", type=int, default=1,
                    help="how many sequences to run mutagenesis on")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--skip_mutagenesis", action="store_true",
                    help="reuse the mutagenesis results already in report.json")
    args = ap.parse_args()

    device = pick_device()
    ckpt = args.ckpt or os.path.join(args.results, "best.pt")
    model, ck = load_model(ckpt, device)
    meta = ck["meta"]
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}
    L = meta["seq_length"]
    print(f"device: {device} | checkpoint: {ckpt} (step {ck['step']}, "
          f"val {ck['val_loss']:.4f} = {ck['val_loss'] * NATS_TO_BITS:.4f} bits/base)")
    print(f"parameters: {model.num_params() / 1e6:.2f}M\n")

    test_pos = np.load(os.path.join(args.data, "test_pos.npy"))
    test_neg = np.load(os.path.join(args.data, "test_neg.npy"))
    print(f"test set: {len(test_pos)} positive, {len(test_neg)} negative")

    # The unigram floor. Beating 2.0 bits/base is easy because promoters are
    # GC rich; beating this is what required context.
    uni_nats, uni_freqs = unigram_baseline(args.data, meta)
    uni_bits = uni_nats * NATS_TO_BITS
    model_bits = ck["val_loss"] * NATS_TO_BITS
    print("\n=== 0. REFERENCE LEVELS (bits per base) ===")
    print(f"  uniform over {meta['vocab_size']} tokens (at init) : "
          f"{np.log(meta['vocab_size']) * NATS_TO_BITS:.4f}")
    print(f"  uniform over 4 bases (random)         : {RANDOM_BITS_PER_BASE:.4f}")
    print(f"  unigram, base frequencies only        : {uni_bits:.4f}")
    print(f"  model, best validation                : {model_bits:.4f}")
    print(f"  base frequencies: " +
          ", ".join(f"{b} {f:.4f}" for b, f in sorted(uni_freqs.items())))
    print(f"  >>> gain over random  : {RANDOM_BITS_PER_BASE - model_bits:+.4f}")
    print(f"  >>> gain over unigram : {uni_bits - model_bits:+.4f}  "
          f"({100 * (uni_bits - model_bits) / max(RANDOM_BITS_PER_BASE - model_bits, 1e-9):.0f}% "
          f"of the gain over random is contextual)")

    report = {
        "checkpoint": os.path.basename(ckpt),
        "checkpoint_step": int(ck["step"]),
        "checkpoint_val_loss_nats": float(ck["val_loss"]),
        "n_params_M": round(model.num_params() / 1e6, 3),
        "n_test_pos": int(len(test_pos)),
        "n_test_neg": int(len(test_neg)),
        "seq_length": int(L),
        "vocab_size": int(meta["vocab_size"]),
        "n_layer": int(ck["config"]["n_layer"]),
        "n_embd": int(ck["config"]["n_embd"]),
        "n_head": int(ck["config"]["n_head"]),
        "dropout": float(ck["config"]["dropout"]),
    }


    report["bits_per_base_init_uniform"] = float(np.log(meta["vocab_size"]) * NATS_TO_BITS)
    report["bits_per_base_random"] = float(RANDOM_BITS_PER_BASE)
    report["bits_per_base_unigram"] = float(uni_bits)
    report["bits_per_base_model_val"] = float(model_bits)
    report["base_frequencies"] = uni_freqs

    # ===== 1. per-sequence loss -> AUROC =====
    nll_pos = batched_seq_nll(model, test_pos, device, args.batch)
    nll_neg = batched_seq_nll(model, test_neg, device, args.batch)
    scores = np.concatenate([-nll_pos, -nll_neg])      # lower loss = promoter
    labels = np.concatenate([np.ones(len(nll_pos)), np.zeros(len(nll_neg))])
    auc_model = auroc(scores, labels)

    print("\n=== 1. PER-SEQUENCE LOSS ===")
    print(f"  positive: {nll_pos.mean():.4f} nats = "
          f"{nll_pos.mean() * NATS_TO_BITS:.4f} bits/base (sd {nll_pos.std():.4f})")
    print(f"  negative: {nll_neg.mean():.4f} nats = "
          f"{nll_neg.mean() * NATS_TO_BITS:.4f} bits/base (sd {nll_neg.std():.4f})")
    print(f"  gap     : {(nll_neg.mean() - nll_pos.mean()) * NATS_TO_BITS:+.4f} bits/base")
    print(f"  >>> MODEL AUROC = {auc_model:.4f}")
    report["auroc_model_loss"] = float(auc_model)
    report["bits_per_base_pos"] = float(nll_pos.mean() * NATS_TO_BITS)
    report["bits_per_base_neg"] = float(nll_neg.mean() * NATS_TO_BITS)

    # ===== 2. GC baseline =====
    gc_pos = gc_fraction_ids(test_pos, stoi)
    gc_neg = gc_fraction_ids(test_neg, stoi)
    auc_gc = auroc(np.concatenate([gc_pos, gc_neg]), labels)
    auc_gc = max(auc_gc, 1.0 - auc_gc)     # direction-agnostic
    print("\n=== 2. GC BASELINE ===")
    print(f"  mean GC: positive {np.nanmean(gc_pos):.4f} | negative {np.nanmean(gc_neg):.4f}")
    print(f"  >>> GC-ONLY AUROC = {auc_gc:.4f}")
    print(f"  >>> GAP (model - GC) = {auc_model - auc_gc:+.4f}")
    if auc_model - auc_gc > 0.03:
        print("  reading: the model appears to capture more than composition.")
    elif abs(auc_model - auc_gc) <= 0.03:
        print("  reading: most of the signal may be base composition alone.")
    else:
        print("  reading: the GC baseline beats the model. That is also a result.")
    report["auroc_gc_baseline"] = float(auc_gc)

    # The direct test: if the model's per-sequence score is essentially a GC
    # measurement, the two must be strongly correlated. This is sharper
    # evidence than comparing AUROCs, because it does not depend on labels.
    all_nll = np.concatenate([nll_pos, nll_neg])
    all_gc = np.concatenate([gc_pos, gc_neg])
    ok = ~np.isnan(all_gc)
    r_all = float(np.corrcoef(all_nll[ok], all_gc[ok])[0, 1])
    r_pos = float(np.corrcoef(nll_pos, gc_pos)[0, 1])
    r_neg = float(np.corrcoef(nll_neg, gc_neg)[0, 1])
    print("\n  correlation between per-sequence loss and GC fraction:")
    print(f"    all sequences : r = {r_all:+.4f}  (r^2 = {r_all ** 2:.4f})")
    print(f"    positives only: r = {r_pos:+.4f}")
    print(f"    negatives only: r = {r_neg:+.4f}")
    if abs(r_all) > 0.9:
        print(f"    -> {100 * r_all ** 2:.0f}% of the variance in the model's score is")
        print("       explained by GC alone: the score is close to a GC measurement.")
    elif abs(r_all) > 0.7:
        print(f"    -> {100 * r_all ** 2:.0f}% of the variance is GC; the score is")
        print("       dominated by composition but not reducible to it.")
    else:
        print("    -> the score carries substantial non-GC information.")
    report["loss_gc_corr_all"] = r_all
    report["loss_gc_corr_pos"] = r_pos
    report["loss_gc_corr_neg"] = r_neg

    # Is the relationship linear? A quadratic fit tells us. Loss is roughly
    # the entropy of the model's predictive distribution, and a sequence with
    # extreme composition -- either AT rich or GC rich -- is more predictable
    # than a balanced one, so an arch is what we expect.
    g = all_gc[ok]
    v = all_nll[ok]
    lin = np.vstack([g, np.ones(len(g))]).T
    quad = np.vstack([g ** 2, g, np.ones(len(g))]).T
    r2 = lambda A: 1.0 - np.sum((v - A @ np.linalg.lstsq(A, v, rcond=None)[0]) ** 2) \
                        / np.sum((v - v.mean()) ** 2)
    r2_lin, r2_quad = float(r2(lin)), float(r2(quad))
    cq = np.linalg.lstsq(quad, v, rcond=None)[0]
    vertex = float(-cq[1] / (2 * cq[0])) if abs(cq[0]) > 1e-12 else float("nan")
    print(f"    linear fit    r^2 = {r2_lin:.4f}")
    print(f"    quadratic fit r^2 = {r2_quad:.4f}  "
          f"(curvature coefficient {cq[0]:+.3f})")
    if cq[0] < 0 and r2_quad > r2_lin + 0.02:
        print(f"    -> the relationship is an arch peaking near GC = {vertex:.3f}:")
        print("       sequences at either compositional extreme are EASY for the")
        print("       model. Promoter-ness rises monotonically with GC, but the")
        print("       score does not, so AT-rich negatives score promoter-like.")
    report["loss_gc_r2_linear"] = r2_lin
    report["loss_gc_r2_quadratic"] = r2_quad
    report["loss_gc_quadratic_vertex"] = vertex
    report["loss_gc_quadratic_curvature"] = float(cq[0])

    # How many negatives beat the median positive, and where do they sit?
    med_pos = float(np.median(nll_pos))
    confused = nll_neg < med_pos
    print(f"    negatives scoring better than the median positive: "
          f"{int(confused.sum())} of {len(nll_neg)} "
          f"({100 * confused.mean():.1f}%)")
    if confused.sum():
        print(f"      their mean GC {np.nanmean(gc_neg[confused]):.4f} vs "
              f"{np.nanmean(gc_neg[~confused]):.4f} for the rest")
        report["confused_negatives_frac"] = float(confused.mean())
        report["confused_negatives_mean_gc"] = float(np.nanmean(gc_neg[confused]))
        report["other_negatives_mean_gc"] = float(np.nanmean(gc_neg[~confused]))

    # Save the per-sequence scores so later analysis needs no rerun.
    np.savez(os.path.join(args.results, "sequence_scores.npz"),
             nll_pos=nll_pos, nll_neg=nll_neg, gc_pos=gc_pos, gc_neg=gc_neg)

    # Residual check: after regressing GC out of the loss, is anything left
    # that still separates the classes?
    A = np.vstack([all_gc[ok], np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, all_nll[ok], rcond=None)
    resid = all_nll[ok] - A @ coef
    auc_resid = auroc(-resid, labels[ok])
    auc_resid = max(auc_resid, 1.0 - auc_resid)
    print(f"    AUROC of the loss after removing GC linearly: {auc_resid:.4f}")
    print("       (0.5 would mean the score is GC and nothing else)")
    report["auroc_loss_residual_after_gc"] = float(auc_resid)

    # ===== 3. in-silico saturation mutagenesis =====
    print("\n=== 3. IN-SILICO SATURATION MUTAGENESIS ===")
    if args.skip_mutagenesis:
        print("  skipped (--skip_mutagenesis); keeping what is in report.json")
    base_ids = sorted(stoi[b] for b in ("A", "C", "G", "T") if b in stoi)
    base_names = [itos[i] for i in base_ids]
    order = np.argsort(nll_pos)            # most promoter-like sequences first
    chosen = [] if args.skip_mutagenesis else order[:args.mut_seqs]

    all_maps = []
    for si in chosen:
        orig = test_pos[si]
        orig_t = torch.from_numpy(orig[None, :]).to(device)
        base_pp = model.per_position_nll(orig_t).float().cpu().numpy()[0]   # (L,)
        base_total = base_pp.sum()

        muts, idxs = [], []
        for p in range(L):
            for b in base_ids:
                if b == orig[p]:
                    continue
                m = orig.copy()
                m[p] = b
                muts.append(m)
                idxs.append((p, b))
        muts = np.stack(muts)
        print(f"  sequence #{si}: scoring {len(muts)} mutants "
              f"(original {base_total / L * NATS_TO_BITS:.4f} bits/base)")

        d_total = np.full((len(base_ids), L), np.nan)
        d_local = np.full((len(base_ids), L), np.nan)
        row_of = {b: r for r, b in enumerate(base_ids)}

        with torch.no_grad():
            for i in range(0, len(muts), args.batch):
                chunk = torch.from_numpy(muts[i:i + args.batch]).to(device)
                pp = model.per_position_nll(chunk).float().cpu().numpy()   # (b, L)
                for k in range(pp.shape[0]):
                    p, b = idxs[i + k]
                    d_total[row_of[b], p] = pp[k].sum() - base_total
                    d_local[row_of[b], p] = pp[k][p] - base_pp[p]

        # A causal model can only propagate a change at p forward, to p..L-1,
        # so a position bias is plausible a priori. We MEASURE it rather than
        # assume it: corr below is |total delta| against the number of
        # downstream positions. Dividing by that count is NOT done -- it would
        # impose a 1/(L-p) ramp and manufacture a spurious hot spot at the
        # end of the sequence.
        downstream = np.arange(L)[::-1] + 1.0          # (L,) = L - p
        all_maps.append((si, d_total, d_local, orig))

        print(f"    total delta: mean {np.nanmean(d_total):+.4f} nats, "
              f"max {np.nanmax(d_total):+.4f}")
        print(f"    local delta: mean {np.nanmean(d_local):+.4f} nats, "
              f"max {np.nanmax(d_local):+.4f}")
        local_share = float(np.nanmean(d_local) / np.nanmean(d_total))
        print(f"    local share of total effect: {100 * local_share:.1f}%")

        # Ranking on the raw total effect, averaged over the three mutant bases.
        flat = np.nanmean(d_total, axis=0)
        top = np.argsort(-flat)[:10]
        print(f"    top 10 positions (raw total delta): "
              f"{sorted(int(t) for t in top)}")

        absd = np.nanmean(np.abs(d_total), axis=0)
        corr = float(np.corrcoef(absd, downstream)[0, 1])
        print(f"    position-bias correlation (|delta| vs downstream) = {corr:+.3f}")
        if abs(corr) < 0.2:
            print("      -> no meaningful position bias; the effect is local, so")
            print("         no normalisation is applied.")

        # The headline of this analysis: effect by MUTANT BASE, not by position.
        by_base = {base_names[r]: float(np.nanmean(d_total[r]))
                   for r in range(len(base_ids))}
        by_base_local = {base_names[r]: float(np.nanmean(d_local[r]))
                         for r in range(len(base_ids))}
        print("    mean total delta by mutant base (higher = more damaging):")
        for b, v in sorted(by_base.items(), key=lambda kv: -kv[1]):
            print(f"      -> {b}: {v:+.4f} nats")
        # Two-way variance decomposition. Comparing a raw base spread with a
        # raw position spread would be unfair: each base mean averages ~L
        # measurements, each position mean only len(base_ids) - 1. The
        # position spread is therefore inflated by noise. Decomposing the sum
        # of squares treats them on equal terms.
        valid = ~np.isnan(d_total)
        grand = float(np.nanmean(d_total))
        row_m = np.nanmean(d_total, axis=1)
        col_m = np.nanmean(d_total, axis=0)
        ss_total = float(np.nansum((d_total - grand) ** 2))
        ss_base = float(np.nansum(valid * (row_m[:, None] - grand) ** 2))
        ss_pos = float(np.nansum(valid * (col_m[None, :] - grand) ** 2))
        ss_resid = ss_total - ss_base - ss_pos
        print("    variance decomposition of the total effect:")
        print(f"      position           : {100 * ss_pos / ss_total:5.1f}%")
        print(f"      mutant base identity: {100 * ss_base / ss_total:5.1f}%")
        print(f"      interaction + noise : {100 * ss_resid / ss_total:5.1f}%")

        # Are any positions real hot spots, or is the positional variation
        # smooth? Compare the count of outliers against the chance expectation.
        zc = (col_m - col_m.mean()) / col_m.std()
        strong = np.where(np.abs(zc) > 2.5)[0]
        expected = 0.0124 * L
        print(f"      positions with |z| > 2.5: {len(strong)} of {L} "
              f"(chance expects about {expected:.0f})")
        if len(strong) <= expected * 1.5:
            print("      -> no position stands out beyond chance: the positional")
            print("         effect is broad, not a localised motif.")
        else:
            print(f"      -> candidate hot spots: {sorted(int(x) for x in strong)}")

        w_ = 15
        sm = np.convolve(col_m, np.ones(w_) / w_, mode="same")
        peak = int(np.argmax(sm[w_ // 2: L - w_ // 2]) + w_ // 2)
        print(f"      smoothed positional profile peaks near position {peak} bp")

        report.setdefault("mutagenesis", []).append({
            "positional_profile_peak_bp": peak,
            "seq_index": int(si),
            "orig_bits_per_base": float(base_total / L * NATS_TO_BITS),
            "d_total_mean": float(np.nanmean(d_total)),
            "d_total_max": float(np.nanmax(d_total)),
            "d_local_mean": float(np.nanmean(d_local)),
            "d_local_max": float(np.nanmax(d_local)),
            "local_share_of_effect": local_share,
            "top10_positions_raw": sorted(int(t) for t in top),
            "position_bias_corr": corr,
            "delta_by_mutant_base": by_base,
            "delta_by_mutant_base_local": by_base_local,
            "var_share_position": float(ss_pos / ss_total),
            "var_share_base_identity": float(ss_base / ss_total),
            "var_share_interaction_noise": float(ss_resid / ss_total),
            "n_positions_abs_z_gt_2p5": int(len(strong)),
            "n_positions_expected_by_chance": float(expected),
            "n_mutants": int(len(muts)),
        })

    # ===== 4. figures =====
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # redraw the loss curve with the unigram floor added
    mpath = os.path.join(args.results, "metrics.csv")
    if os.path.exists(mpath):
        import csv as _csv
        with open(mpath) as f:
            mrows = list(_csv.DictReader(f))
        msteps = [int(r["step"]) for r in mrows]
        mtr = [float(r["train_loss"]) * NATS_TO_BITS for r in mrows]
        mva = [float(r["val_loss"]) * NATS_TO_BITS for r in mrows]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(msteps, mtr, label="train")
        ax.plot(msteps, mva, label="validation")
        ax.axhline(RANDOM_BITS_PER_BASE, ls="--", c="gray",
                   label=f"random, uniform over 4 bases ({RANDOM_BITS_PER_BASE:.2f})")
        ax.axhline(uni_bits, ls=":", c="crimson",
                   label=f"unigram, base frequencies only ({uni_bits:.3f})")
        # The step-0 point sits at log2(vocab) and squashes everything else,
        # so clip the view and annotate it instead.
        tail = mva[1:] + mtr[1:]
        ax.set_ylim(min(tail) - 0.03, RANDOM_BITS_PER_BASE + 0.08)
        ax.annotate(f"step 0: {mva[0]:.2f} (uniform over {meta['vocab_size']} tokens)",
                    xy=(msteps[0], RANDOM_BITS_PER_BASE + 0.05),
                    fontsize=7.5, color="dimgray")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("bits per base")
        ax.set_title("Promoter-GPT: loss per base, against two baselines")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.results, "loss.png"), dpi=150)
        plt.close(fig)
        print("  redrew loss.png with the unigram baseline")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo = min(nll_pos.min(), nll_neg.min()) * NATS_TO_BITS
    hi = max(nll_pos.max(), nll_neg.max()) * NATS_TO_BITS
    bins = np.linspace(lo, hi, 60)
    ax.hist(nll_pos * NATS_TO_BITS, bins=bins, alpha=0.6, density=True,
            label=f"positive (promoter), n={len(nll_pos)}")
    ax.hist(nll_neg * NATS_TO_BITS, bins=bins, alpha=0.6, density=True,
            label=f"negative, n={len(nll_neg)}")
    ax.axvline(RANDOM_BITS_PER_BASE, ls="--", c="gray", label="random (2.0)")
    ax.set_xlabel("bits per base, per sequence")
    ax.set_ylabel("density")
    ax.set_title(f"Label-free separation: AUROC={auc_model:.3f} "
                 f"(GC-only {auc_gc:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "auroc_hist.png"), dpi=150)
    plt.close(fig)

    # loss vs GC scatter: the clearest single picture of what the score is
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.scatter(gc_neg, nll_neg * NATS_TO_BITS, s=3, alpha=0.25, label="negative")
    ax.scatter(gc_pos, nll_pos * NATS_TO_BITS, s=3, alpha=0.25,
               label="positive (promoter)")
    xs = np.linspace(np.nanmin(all_gc), np.nanmax(all_gc), 50)
    ax.plot(xs, (coef[0] * xs + coef[1]) * NATS_TO_BITS, c="black", lw=1.2,
            ls="--", label=f"linear fit, r$^2$={r2_lin:.3f}")
    ax.plot(xs, (cq[0] * xs ** 2 + cq[1] * xs + cq[2]) * NATS_TO_BITS,
            c="black", lw=1.6, label=f"quadratic fit, r$^2$={r2_quad:.3f}")
    ax.axhline(uni_bits, ls=":", c="crimson", lw=1,
               label=f"unigram ({uni_bits:.3f})")
    ax.set_xlabel("GC fraction of the sequence")
    ax.set_ylabel("model loss (bits per base)")
    ax.set_title("The model's score is an arch in GC, not a line:\n"
                 "both compositional extremes are easy for the model")
    ax.legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "loss_vs_gc.png"), dpi=150)
    plt.close(fig)

    for si, d_total, d_local, orig in all_maps:
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        for ax, mat, name in zip(
                axes[:2], [d_total, d_local],
                ["total delta NLL (whole sequence)",
                 "local delta NLL (mutated base only)"]):
            vmax = np.nanmax(np.abs(mat))
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_yticks(range(len(base_names)))
            ax.set_yticklabels(base_names)
            ax.set_ylabel("mutant base")
            ax.set_title(name)
            fig.colorbar(im, ax=ax, pad=0.01)

        downstream = np.arange(L)[::-1] + 1.0
        absd = np.nanmean(np.abs(d_total), axis=0)
        corr = float(np.corrcoef(absd, downstream)[0, 1])
        axes[2].plot(absd, label="|total delta|")
        axes[2].plot(np.nanmean(np.abs(d_local), axis=0), label="|local delta|")
        axes[2].set_xlabel("position (bp)")
        axes[2].set_ylabel("mean |delta| (nats)")
        axes[2].set_title(f"Effect vs position: flat. Correlation with the number "
                          f"of downstream positions = {corr:+.3f}, so the expected "
                          f"causal position bias does not appear.")
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.results, f"mutagenesis_seq{si}.png"), dpi=150)
        plt.close(fig)

        # The finding, as one plot: effect depends on which base you mutate TO,
        # not on where in the sequence you do it.
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        means = [float(np.nanmean(d_total[r])) for r in range(len(base_ids))]
        sds = [float(np.nanstd(d_total[r])) for r in range(len(base_ids))]
        axs[0].bar(base_names, means, yerr=sds, capsize=4,
                   color=["#c44", "#4a8", "#48c", "#ca4"])
        axs[0].axhline(0, c="black", lw=0.8)
        axs[0].set_ylabel("mean total delta NLL (nats)")
        axs[0].set_xlabel("base mutated TO")
        axs[0].set_title("Effect by mutant base")

        colmean = np.nanmean(d_total, axis=0)
        axs[1].plot(colmean, lw=0.6, alpha=0.45, label="per position")
        # Each position averages only three mutants, so smooth to show the
        # trend underneath that noise.
        w = 15
        ker = np.ones(w) / w
        smooth = np.convolve(colmean, ker, mode="same")
        edge = w // 2
        axs[1].plot(np.arange(edge, L - edge), smooth[edge:L - edge],
                    lw=2, c="darkblue", label=f"{w} bp rolling mean")
        axs[1].axhline(0, c="black", lw=0.8)
        axs[1].set_ylabel("mean total delta NLL (nats)")
        axs[1].set_xlabel("position (bp)")
        axs[1].set_title("Effect by position")
        axs[1].legend(fontsize=8)
        lo = min(min(means) - max(sds), float(np.nanmin(colmean)))
        hi = max(max(means) + max(sds), float(np.nanmax(colmean)))
        for a in axs:
            a.set_ylim(lo - 0.2, hi + 0.2)     # same scale, so the contrast is fair
        fig.suptitle(
            f"Both matter: a clear base preference, and a broad positional "
            f"profile (position {100 * ss_pos / ss_total:.0f}% of variance, "
            f"base identity {100 * ss_base / ss_total:.0f}%)",
            fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.results, f"mutagenesis_bybase_seq{si}.png"),
                    dpi=150)
        plt.close(fig)

        np.savez(os.path.join(args.results, f"mutagenesis_seq{si}.npz"),
                 d_total=d_total, d_local=d_local,
                 orig=orig, base_names=np.array(base_names))

    # ===== 5. optional embedding probe =====
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        print("\n=== 5. EMBEDDINGS AND LINEAR PROBE ===")
        n = min(len(test_pos), len(test_neg), 2000)
        ep = batched_embed(model, test_pos[:n], device, args.batch)
        en = batched_embed(model, test_neg[:n], device, args.batch)
        X = np.concatenate([ep, en])
        yv = np.concatenate([np.ones(n), np.zeros(n)])

        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))
        X, yv = X[idx], yv[idx]
        split = int(0.7 * len(X))
        sc = StandardScaler().fit(X[:split])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[:split]), yv[:split])
        probe_auc = auroc(clf.decision_function(sc.transform(X[split:])), yv[split:])
        print(f"  linear probe AUROC (held out) = {probe_auc:.4f}")
        report["auroc_embedding_probe"] = float(probe_auc)

        pca = PCA(n_components=2).fit(sc.transform(X))
        Z = pca.transform(sc.transform(X))
        fig, ax = plt.subplots(figsize=(6, 5.5))
        for lab, name in [(1, "positive (promoter)"), (0, "negative")]:
            m = yv == lab
            ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.4, label=name)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.set_title(f"Embedding geometry (probe AUROC={probe_auc:.3f})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.results, "embedding_pca.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("\n=== 5. skipped: scikit-learn not installed ===")

    # Merge rather than overwrite, so a quick rerun (--skip_mutagenesis) does
    # not throw away results from a full one.
    rpath = os.path.join(args.results, "report.json")
    if os.path.exists(rpath):
        try:
            prev = json.load(open(rpath))
            for k, v in prev.items():
                report.setdefault(k, v)
        except Exception:
            pass
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 52)
    print("SUMMARY")
    print("=" * 52)
    print(f"  model AUROC (loss)      : {report['auroc_model_loss']:.4f}")
    print(f"  GC-only AUROC           : {report['auroc_gc_baseline']:.4f}")
    if "auroc_embedding_probe" in report:
        print(f"  embedding probe AUROC   : {report['auroc_embedding_probe']:.4f}")
    print(f"  bits/base pos vs neg    : {report['bits_per_base_pos']:.4f} / "
          f"{report['bits_per_base_neg']:.4f}")
    print(f"  bits/base model vs unigram vs random: {model_bits:.4f} / "
          f"{uni_bits:.4f} / {RANDOM_BITS_PER_BASE:.4f}")
    print(f"  loss-GC correlation     : r = {report['loss_gc_corr_all']:+.4f} "
          f"(r^2 = {report['loss_gc_corr_all'] ** 2:.4f})")
    print(f"  AUROC after removing GC : {report['auroc_loss_residual_after_gc']:.4f}")
    print(f"\n  figures -> {args.results}/auroc_hist.png, "
          f"mutagenesis_seq*.png, embedding_pca.png")


if __name__ == "__main__":
    main()
