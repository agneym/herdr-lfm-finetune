#!/usr/bin/env python3
"""train_herdr_agent.py — train the Herdr Needle 2 LoRA adapter.

Portable: works on a GPU box (Google Colab, RunPod, a local CUDA machine) or CPU.
Installs nothing; requires `cactus-needle[gpu]` (or [metal]/cpu) already present.

Usage
    python train_herdr_agent.py data.jsonl --epochs 12 --out adapter.pkl
    python train_herdr_agent.py data.jsonl --batch-size 4 --max-len 4096 --epochs 10

Defaults are tuned for the full 25-tool Herdr catalogue (seq_len 4096) on a
single T4: batch 2 (safe under 16GB VRAM), 12 epochs.  On CPU it drops to batch 1.
After training, build the tuned archive on the target machine:

    needle build checkpoints/needle2.pkl --lora adapter.pkl --out tuned.cact
"""
import argparse
import shlex
import shutil
import subprocess
import sys


def detect_backend():
    try:
        import jax
        return jax.default_backend().lower()
    except Exception:
        return "unknown"


def default_batch(backend):
    if backend in ("gpu", "cuda", "rocm"):
        return 2
    return 1


def main():
    ap = argparse.ArgumentParser(description="Train the Herdr Needle LoRA adapter.")
    ap.add_argument("data", help="path to data.jsonl")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default: 2 on GPU, 1 on CPU")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", default="adapter.pkl")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--base", default="checkpoints/needle2.pkl")
    args = ap.parse_args()

    backend = detect_backend()
    batch = args.batch_size or default_batch(backend)
    needle_bin = shutil.which("needle")
    if not needle_bin:
        sys.exit("`needle` CLI not found on PATH - install cactus-needle first")

    cmd = [needle_bin, "finetune", args.data,
           "--epochs", str(args.epochs),
           "--batch-size", str(batch),
           "--max-len", str(args.max_len),
           "--out", args.out,
           "--lora-rank", str(args.lora_rank),
           "--lora-alpha", str(args.lora_alpha),
           "--lr", str(args.lr),
           "--val-split", str(args.val_split),
           "--checkpoint", args.base]
    print(f"backend: {backend}   batch-size: {batch}   epochs: {args.epochs}   max-len {args.max_len}")
    print("running: " + " ".join(shlex.quote(c) for c in cmd))
    print()
    subprocess.run(cmd, check=False)
    print()
    print("training finished. Steps to finish on your machine:")
    print(f"  needle build {args.base} --lora {args.out} --out tuned.cact")
    print("  python ask_herdr.py --weights tuned.cact --query \"<your request>\"")


if __name__ == "__main__":
    main()
