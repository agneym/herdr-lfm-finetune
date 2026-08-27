#!/usr/bin/env python3
"""Pin the current eval holdout to a JSON file keyed by query string.

The holdout is computed exactly as eval_lfm2.py / train_lfm2.py compute it
(seed 42, last N% after a full shuffle). Persisting it lets later dataset
versions append training rows without shifting the eval set — pass the file
back with --holdout to eval_lfm2.py and train_lfm2.py.

Usage:
  python pin_holdout.py --data dataset.jsonl --out runs/eval_v5_holdout.json
"""
import argparse
import json

from split import save_pinned_holdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.jsonl")
    ap.add_argument("--out", default="runs/eval_holdout.json")
    ap.add_argument("--split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    idx = save_pinned_holdout(args.out, rows, args.split, args.seed)
    print(f"pinned {len(idx)} rows to {args.out} "
          f"(n={len(rows)}, split={args.split}, seed={args.seed})")


if __name__ == "__main__":
    main()
