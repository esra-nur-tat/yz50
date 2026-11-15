"""Download and encode the promoter dataset.

Pulls Genomic Benchmarks `human_nontata_promoters`, splits positives
(promoters) from negatives, and saves single-base encodings as .npy.

Encoding: 0 = BOS (added during training), 1..K = observed bases, sorted.

Usage:
    python3 prepare.py
    python3 prepare.py --peek     # download and report only, write nothing
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

HF_REPO = "katarinagresova/Genomic_Benchmarks_human_nontata_promoters"
OUT_DIR = "data"


def load_raw():
    """Load from the Hub, falling back to parquet if `datasets` fails."""
    try:
        from datasets import load_dataset
        print(f"loading with datasets: {HF_REPO}")
        ds = load_dataset(HF_REPO)
        return {split: ds[split] for split in ds.keys()}
    except Exception as exc:
        print(f"datasets path failed: {exc}")
        print("falling back to parquet...")
        import pandas as pd
        base = f"hf://datasets/{HF_REPO}"
        out = {}
        for split in ("train", "test"):
            out[split] = pd.read_parquet(f"{base}/{split}.parquet")
        return out


def detect_columns(split_obj):
    """Find the sequence and label columns by content, not by name."""
    if hasattr(split_obj, "column_names"):
        cols = list(split_obj.column_names)
        row = split_obj[0]
    else:
        cols = list(split_obj.columns)
        row = split_obj.iloc[0].to_dict()

    seq_col, label_col = None, None
    for c in cols:
        v = row[c]
        if isinstance(v, str) and len(v) > 20 and set(v.upper()) <= set("ACGTN"):
            seq_col = c
        elif isinstance(v, (int, np.integer, bool, np.bool_)):
            label_col = c
    if seq_col is None:
        raise SystemExit(f"No sequence column found. Columns: {cols}, row: {row}")
    if label_col is None:
        raise SystemExit(f"No label column found. Columns: {cols}, row: {row}")
    print(f"  columns: sequence='{seq_col}', label='{label_col}'  (all: {cols})")
    return seq_col, label_col


def to_lists(split_obj, seq_col, label_col):
    if hasattr(split_obj, "column_names"):
        return list(split_obj[seq_col]), list(split_obj[label_col])
    return split_obj[seq_col].tolist(), split_obj[label_col].tolist()


def gc_fraction(seq):
    s = seq.upper()
    usable = sum(1 for ch in s if ch in "ACGT")
    if usable == 0:
        return float("nan")
    return sum(1 for ch in s if ch in "GC") / usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", action="store_true", help="download, report, write nothing")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    splits = load_raw()
    print(f"\nsplits: {list(splits.keys())}")

    data = {}             # (split, label) -> [sequence]
    alphabet = Counter()
    lengths = Counter()

    for split, obj in splits.items():
        seq_col, label_col = detect_columns(obj)
        seqs, labels = to_lists(obj, seq_col, label_col)
        print(f"  {split}: {len(seqs)} sequences")
        for s, y in zip(seqs, labels):
            s = s.upper()
            alphabet.update(s)
            lengths[len(s)] += 1
            data.setdefault((split, int(y)), []).append(s)

    print("\n=== ALPHABET ===")
    total_bases = sum(alphabet.values())
    for ch, n in alphabet.most_common():
        print(f"  {ch}: {n:>12,}  ({100 * n / total_bases:.4f}%)")

    print("\n=== LENGTH DISTRIBUTION ===")
    for L, n in lengths.most_common(8):
        print(f"  {L} bp: {n:,} sequences")
    modal_len = lengths.most_common(1)[0][0]
    off = sum(n for L, n in lengths.items() if L != modal_len)
    print(f"  -> modal length {modal_len} bp; {off} sequences of other lengths")

    print("\n=== CLASS BALANCE AND GC ===")
    for key in sorted(data.keys()):
        split, label = key
        seqs = data[key]
        gcs = np.array([gc_fraction(s) for s in seqs])
        name = "positive (promoter)" if label == 1 else "negative"
        print(f"  {split:>6} / {name:<20} n={len(seqs):>7,}  "
              f"GC mean={np.nanmean(gcs):.4f}  sd={np.nanstd(gcs):.4f}")

    tr_pos = data.get(("train", 1), [])
    tr_neg = data.get(("train", 0), [])
    if tr_pos and tr_neg:
        gp = np.nanmean([gc_fraction(s) for s in tr_pos])
        gn = np.nanmean([gc_fraction(s) for s in tr_neg])
        print(f"\n  GC gap (pos - neg) = {gp - gn:+.4f}")
        print("  NOTE: a large gap means the GC-only baseline will score well.")
        print("        Whether the model learns more than that is the question.")

    if args.peek:
        print("\n--peek given, nothing written.")
        return

    bases = sorted(alphabet.keys())
    stoi = {ch: i + 1 for i, ch in enumerate(bases)}   # 0 is BOS
    itos = {i + 1: ch for i, ch in enumerate(bases)}
    vocab_size = len(bases) + 1
    print("\n=== ENCODING ===")
    print("  BOS=0, " + ", ".join(f"{ch}={i}" for ch, i in stoi.items()))
    print(f"  vocab_size = {vocab_size}")

    os.makedirs(args.out, exist_ok=True)
    manifest = {}

    for (split, label), seqs in sorted(data.items()):
        keep = [s for s in seqs if len(s) == modal_len]
        dropped = len(seqs) - len(keep)
        arr = np.zeros((len(keep), modal_len), dtype=np.uint8)
        for i, s in enumerate(keep):
            arr[i] = [stoi[ch] for ch in s]
        tag = "pos" if label == 1 else "neg"
        fname = f"{split}_{tag}.npy"
        np.save(os.path.join(args.out, fname), arr)
        manifest[f"{split}_{tag}"] = {
            "file": fname, "n": int(arr.shape[0]),
            "length": int(modal_len), "dropped_offlength": int(dropped),
        }
        print(f"  wrote {fname}: {arr.shape}  ({dropped} dropped)")

    meta = {
        "repo": HF_REPO,
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "seq_length": int(modal_len),
        "bos_id": 0,
        "splits": manifest,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nwrote meta.json. Training uses train_pos.npy only.")


if __name__ == "__main__":
    main()
