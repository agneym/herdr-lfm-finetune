#!/usr/bin/env python3
"""export_gguf.py — merge the LoRA adapter into the base and export a GGUF for llama.cpp.

Mirrors the validated chain (merge -> convert_hf_to_gguf.py -> llama-quantize)
so a fresh clone can reproduce the published GGUF from the adapter alone.

Steps
-----
1. Merge `base + adapter` -> `--out-dir/merged_hf/` (bf16 safetensors, cached).
2. `convert_hf_to_gguf.py` -> `--out-dir/lfm2-herdr-{outtype}.gguf`.
3. `llama-quantize` -> one GGUF per `--quant` value.
4. `--push` uploads the F16 + quantized GGUFs (and an auto model card) to `--repo`.

Requirements
------------
- A llama.cpp checkout with the converter and a built `llama-quantize`:
      git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
      cmake -B /tmp/llama.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
      cmake --build /tmp/llama.cpp/build --target llama-quantize
  Point `--llama-cpp-dir` at it (or set LLAMA_CPP_DIR). The converter uses the
  repo's bundled `gguf-py` and needs `transformers`/`torch` (already in .venv).
- For `--push`, be logged in to Hugging Face (cached token or HF_TOKEN).

Examples
--------
  .venv/bin/python scripts/export_gguf.py \
      --adapter adapters/lfm2_herdr_lora --quant Q4_K_M --quant Q5_K_M --push
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing dependency: {e}\ninstall with: .venv/bin/pip install transformers peft torch")

DEFAULT_REPO = "agney/lfm2-herdr-gguf"
DEFAULT_BASE = "LiquidAI/LFM2-350M"
DEFAULT_QUANTS = ["Q4_K_M", "Q5_K_M"]


def log(*a):
    print("[export_gguf]", *a, flush=True)


def run(cmd: list[str], cwd: Path | None = None):
    log("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def merge(base: str, adapter: str, out_dir: Path, force: bool) -> Path:
    merged = out_dir / "merged_hf"
    if (merged / "model.safetensors").exists() and not force:
        log(f"reusing cached merged model at {merged}")
        return merged
    log(f"merging {adapter} into {base} -> {merged}")
    merged.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(merged, safe_serialization=True)
    tok.save_pretrained(merged)
    return merged


def llama_cpp_dir(llama: Path):
    convert = llama / "convert_hf_to_gguf.py"
    if not convert.exists():
        sys.exit(
            f"convert_hf_to_gguf.py not found in {llama}\n"
            "clone it: git clone --depth 1 https://github.com/ggml-org/llama.cpp.git <llama-cpp-dir>"
        )
    return convert


def convert(convert: Path, merged: Path, out_dir: Path, outtype: str) -> Path:
    out = out_dir / f"lfm2-herdr-{outtype}.gguf"
    log(f"converting {merged} -> {out} ({outtype})")
    run([sys.executable, convert, merged, "--outfile", out, "--outtype", outtype])
    return out


def quantize(llama: Path, f16_gguf: Path, out_dir: Path, quants: list[str]):
    qbin = llama / "build" / "bin" / "llama-quantize"
    if not qbin.exists():
        sys.exit(
            f"llama-quantize not built at {qbin}\n"
            "build it: cmake -B <llama-cpp-dir>/build -G Ninja "
            "-DCMAKE_BUILD_TYPE=Release && cmake --build <llama-cpp-dir>/build --target llama-quantize"
        )
    for q in quants:
        out = out_dir / f"lfm2-herdr-{q}.gguf"
        run([qbin, f16_gguf, out, q])


def push(repo: str, out_dir: Path, quants: list[str], outtype: str):
    from huggingface_hub import HfApi
    api = HfApi()
    files = [out_dir / f"lfm2-herdr-{outtype}.gguf"] + [
        out_dir / f"lfm2-herdr-{q}.gguf" for q in quants
    ]
    card = out_dir / "README.md"
    if not card.exists():
        card.write_text(model_card(repo, quants, outtype))
        files.append(card)
    log(f"pushing to {repo}: {[f.name for f in files]}")
    # Upload only the GGUFs + card; the merged bf16 dir lives under out_dir too
    # and must not be published to the GGUF repo.
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    for f in files:
        if f.exists():
            api.upload_file(path_or_fileobj=f, path_in_repo=f.name, repo_id=repo, repo_type="model")


def model_card(repo: str, quants: list[str], outtype: str) -> str:
    qs = " | ".join(quants)
    return f"""---
base_model: LiquidAI/LFM2-350M
language: en
license: mit
tags: [lfm2, llama.cpp, herdr, tool-calling, gguf]
---
# Herdr expert — LFM2-350M (GGUF)

Merged LoRA fine-tune of **LiquidAI/LFM2-350M** ([adapter: agney/lfm2-herdr-lora](https://huggingface.co/agney/lfm2-herdr-lora))
exported to GGUF for **llama.cpp** — no Python needed to run.

Quantizations: **{qs}** (floor {outtype} = `lfm2-herdr-{outtype}.gguf`).

## Run (llama.cpp)

```sh
llama-cli -m lfm2-herdr-{quants[0]}.gguf -p "<rendered prompt>" -st -n 128 --temp 0
```

The model answers in native `<|tool_call_start|>[name(k=v, ...)]<|tool_call_end|>` syntax.
Feed the system env + Herdr tool schemas the same way the training prompt did
(see the repo's `eval_lfm2.parse_calls` for the parser). Narrow specialist: 25 Herdr ops
and off-topic refusal only.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default="adapters/lfm2_herdr_lora")
    ap.add_argument("--out-dir", default="runs/export")
    ap.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR", "/tmp/llama.cpp"))
    ap.add_argument("--outtype", default="f16")
    ap.add_argument("--quant", action="append", default=DEFAULT_QUANTS)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--push", action="store_true", help="upload GGUF to Hugging Face Hub")
    ap.add_argument("--force", action="store_true", help="re-merge even if cached")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    llama = Path(args.llama_cpp_dir).resolve()

    merged = merge(args.base, args.adapter, out_dir, args.force)
    converter = llama_cpp_dir(llama)
    if args.quant:
        f16 = convert(converter, merged, out_dir, args.outtype)
        quantize(llama, f16, out_dir, args.quant)
    if args.push:
        push(args.repo, out_dir, args.quant, args.outtype)


if __name__ == "__main__":
    main()
