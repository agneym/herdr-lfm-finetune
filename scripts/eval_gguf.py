#!/usr/bin/env python3
"""eval_gguf.py — score a llama.cpp GGUF of the Herdr expert on the pinned
holdout, mirroring eval_lfm2.py so GGUF results compare head-to-head with the
transformers/PEFT numbers.

The prompt is rendered with the HF tokenizer (same as eval_lfm2.py), then run
through llama.cpp. Two backends:

  1. llama-server native /completion (default) — loads the model once, batches
     all rows, fast. `--spawn` starts+stops the server automatically, or point
     `--server-url` at a running one.
  2. llama-cli single-turn per row (`--cli`) — no server, slower.

Reuses eval_lfm2.parse_calls / normalize_call / norm_calls for identical scoring.

Examples
--------
  # bring up a server on :8080 and eval Q4_K_M
  .venv/bin/python scripts/eval_gguf.py --gguf runs/export/lfm2-herdr-Q4_K_M.gguf \
      --holdout runs/results/eval_v8_holdout.json --spawn

  # against an already-running llama-server
  .venv/bin/python scripts/eval_gguf.py --gguf f.gguf --holdout h.json \
      --server-url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

# The repo's parse/scoring helpers live in the repo root, not scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the exact parser + scoring from the transformers eval.
import eval_lfm2 as E  # parse_calls, normalize_call, norm_calls, names_of, arg_key

DATA = "dataset.jsonl"
MODEL_ID = "LiquidAI/LFM2-350M"
DEFAULT_QUANT = "Q4_K_M"


def render_prompt(tok, row, tools):
    return tok.apply_chat_template(
        row["messages"][:-1], tools=tools, tokenize=False, add_generation_prompt=True)


class CompletionServer:
    """Thin wrapper over llama-server's native /completion endpoint."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def generate(self, prompt: str, max_new: int, timeout: int = 120) -> str:
        body = json.dumps({
            "prompt": prompt,
            "n_predict": max_new,
            "temperature": 0,
            "repeat_penalty": 1.05,
            "min_p": 0.15,
            "cache_prompt": True,
        }).encode()
        req = urllib.request.Request(
            self.url + "/completion", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)["content"]


def spawn_server(gguf: str, port: int = 8080, llama_dir: str = "/tmp/llama.cpp") -> subprocess.Popen:
    bin_ = os.path.join(llama_dir, "build", "bin", "llama-server")
    if not os.path.exists(bin_):
        sys.exit(f"llama-server not found at {bin_}; build it: cmake --build {llama_dir}/build --target llama-server")
    proc = subprocess.Popen(
        [bin_, "-m", gguf, "--port", str(port), "--threads", str(os.cpu_count())],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    # wait for health
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(120):
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    sys.exit("llama-server did not become healthy")


def generate_cli(prompt: str, gguf: str, max_new: int, llama_dir: str) -> str:
    bin_ = os.path.join(llama_dir, "build", "bin", "llama-cli")
    if not os.path.exists(bin_):
        sys.exit(f"llama-cli not found at {bin_}")
    p = subprocess.run(
        [bin_, "-m", gguf, "-p", prompt, "-st", "-n", str(max_new), "--temp", "0",
         "--repeat-penalty", "1.05", "--no-display-prompt", "-t", str(os.cpu_count())],
        capture_output=True, text=True, timeout=600)
    return p.stdout


def score_rows(rows, eval_run, generate, max_new, detail_limit=10):
    exact = exact_norm = tool_ok = n_expected = n_offtopic = off_ok = 0
    arg_counts = {}
    detail = []
    for i in eval_run:
        row = rows[i]
        expected = row["expected"]
        try:
            text = generate(render_prompt(tok, row, row.get("tools") or SCHEMAS), max_new)
        except Exception as exc:  # noqa: BLE001
            detail.append((i, row["messages"][1]["content"], expected, [], f"ERROR: {exc!r}"))
            continue
        predicted = E.parse_calls(text)
        normalized = [E.normalize_call(c) for c in predicted]
        pn, en = E.norm_calls(predicted), E.norm_calls(expected)
        pn_norm = E.norm_calls(normalized)
        pw, ew = E.names_of(predicted), E.names_of(expected)
        if en:
            n_expected += 1
            exact += (pn == en)
            exact_norm += (pn_norm == en)
            tool_ok += (pw == ew)
            for name in ew:
                ok = E.arg_key(name, predicted) == E.arg_key(name, expected)
                rec = arg_counts.setdefault(name, [0, 0])
                rec[0] += ok
                rec[1] += 1
        else:
            n_offtopic += 1
            off_ok += (not predicted)
            if predicted:
                detail.append((i, row["messages"][1]["content"], [], predicted, "OFF-TOPIC but called a tool"))
                continue
        if en and pn != en:
            detail.append((i, row["messages"][1]["content"], expected, predicted, ""))
    return exact, exact_norm, tool_ok, n_expected, n_offtopic, off_ok, arg_counts, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True, help="path to the .gguf to score")
    ap.add_argument("--holdout", required=True, help="pinned holdout JSON")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0, help="eval only the first N rows (debug)")
    ap.add_argument("--detail-limit", type=int, default=10, help="max mismatches to print")
    ap.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR", "/tmp/llama.cpp"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--server-url", help="use a running llama-server at this URL")
    g.add_argument("--spawn", action="store_true", help="start+stop a temporary llama-server")
    g.add_argument("--cli", action="store_true", help="use llama-cli single-turn (no server)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    from herdr_tools import SCHEMAS
    from split import load_pinned_holdout
    from transformers import AutoTokenizer

    global tok
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    rows = [json.loads(l) for l in open(args.data)]
    eval_idx, hmeta = load_pinned_holdout(args.holdout, rows)
    eval_run = eval_idx[: args.limit] if args.limit else eval_idx
    print(f"holdout: {args.holdout} (pinned {len(eval_idx)} rows)   gguf: {args.gguf}")
    print(f"data rows: {len(rows)}   eval rows: {len(eval_run)}")

    spawned = None
    if args.server_url:
        gen = CompletionServer(args.server_url).generate
    elif args.spawn:
        spawned = spawn_server(args.gguf, args.port, args.llama_cpp_dir)
        gen = CompletionServer(f"http://127.0.0.1:{args.port}").generate
    elif args.cli:
        gen = lambda p, n: generate_cli(p, args.gguf, n, args.llama_cpp_dir)
    else:
        sys.exit("pick one backend: --server-url, --spawn, or --cli")

    try:
        exact, exact_norm, tool_ok, n_expected, n_offtopic, off_ok, arg_counts, detail = score_rows(
            rows, eval_run, gen, args.max_new_tokens, detail_limit=args.detail_limit)
    finally:
        if spawned is not None:
            spawned.terminate()
            spawned.wait(timeout=10)

    print("\n== summary ==")
    print(f"  eval rows            : {len(eval_run)}")
    if n_expected:
        print(f"  exact-call accuracy  : {exact}/{n_expected} = {100*exact/n_expected:.1f}%")
        print(f"  exact (normalized)   : {exact_norm}/{n_expected} = {100*exact_norm/n_expected:.1f}%")
        print(f"  tool-selection acc   : {tool_ok}/{n_expected} = {100*tool_ok/n_expected:.1f}%")
    if n_offtopic:
        print(f"  off-topic (no call)  : {off_ok}/{n_offtopic} = {100*off_ok/n_offtopic:.1f}%")
    print("\n== per-tool argument grounding ==")
    for name in sorted(arg_counts):
        ok, total = arg_counts[name]
        print(f"  {name:<22} {ok}/{total} exact ({100*ok/total:.0f}%)")
    if detail:
        print(f"\n== mismatches (first {min(args.detail_limit, len(detail))}) ==")
        for i, q, exp, pred, note in detail[:args.detail_limit]:
            print(f"  row {i}: {q[:80]}")
            if note:
                print(f"    {note}")
            print(f"    expected : {json.dumps(exp)[:150]}")
            print(f"    predicted: {json.dumps(pred)[:150]}")


if __name__ == "__main__":
    main()
