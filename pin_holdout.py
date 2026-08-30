#!/usr/bin/env python3
"""Pin the current eval holdout to a JSON file keyed by query string.

The holdout is computed exactly as eval_lfm2.py / train_lfm2.py compute it
(seed 42, last N% after a full shuffle). Persisting it lets later dataset
versions append training rows without shifting the eval set — pass the file
back with --holdout to eval_lfm2.py and train_lfm2.py.

Usage:
  python pin_holdout.py --data dataset.jsonl --out runs/results/eval_v8_holdout.json

  --out is REQUIRED: never let a re-pin clobber the 98-row eval_v5_holdout.json
  that the published numbers are scored against. Add --force to overwrite an
  existing pin (intended for a brand-new versioned holdout, not the live one).
"""
import argparse
import json
import os

from split import save_pinned_holdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.jsonl")
    ap.add_argument("--out", required=True,
                    help="path to write the pinned holdout JSON (required)")
    ap.add_argument("--split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing holdout file")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        ap.error(f"{args.out} already exists; pass --force to overwrite it. "
                 "Never reuse the live 98-row eval_v5_holdout.json.")

    rows = [json.loads(l) for l in open(args.data)]
    idx = save_pinned_holdout(args.out, rows, args.split, args.seed)
    print(f"pinned {len(idx)} rows to {args.out} "
          f"(n={len(rows)}, split={args.split}, seed={args.seed})")


if __name__ == "__main__":
    main()
