#!/usr/bin/env python3
"""fetch_adapter.py — download the tuned LoRA adapter from Hugging Face Hub.

The trainer weights are NOT committed to git; they live on the Hub. This fills
adapters/lfm2_herdr_lora/ so `make eval` and the README quick-start work on a
fresh clone.

Public model repos need no token. For a private repo, export HF_TOKEN (or
HUGGING_FACE_HUB_TOKEN) in the environment.

Usage:
  .venv/bin/python scripts/fetch_adapter.py                # defaults: agney/lfm2-herdr-lora
  .venv/bin/python scripts/fetch_adapter.py --repo <user>/<repo> --out adapters/lfm2_herdr_lora
"""
import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = os.environ.get("HF_ADAPTER_REPO", "agney/lfm2-herdr-lora")
DEFAULT_OUT = "adapters/lfm2_herdr_lora"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="HF model repo id (default: %s)" % DEFAULT_REPO)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="destination adapter dir (default: %s)" % DEFAULT_OUT)
    args = ap.parse_args()

    print(f"downloading adapter {args.repo} -> {args.out} ...")
    src = Path(snapshot_download(args.repo))
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        # Skip Hub-managed files that don't belong in the local adapter dir.
        if rel.name in (".gitattributes", ".gitignore"):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        n += 1
    print(f"fetched {n} files into {dst}/")


if __name__ == "__main__":
    main()
