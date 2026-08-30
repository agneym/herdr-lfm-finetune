#!/usr/bin/env python3
"""herdr_gguf.py — turn a natural-language Herdr request into a tool call,
running the GGUF expert through llama.cpp.

The *model* runs as a single binary (no Python model-loading). This script only
(1) renders the prompt (system env + Herdr tool schemas) the same way the
training prompt did, (2) sends it to llama.cpp, and (3) parses the native
<|tool_call_start|> answer. It is the "don't hand-build the prompt" helper.

Backends
--------
  --server-url <url>   query a running llama-server (native /completion)
  --spawn              start + stop a temporary llama-server for this query
  --cli                use llama-cli single-turn (no server, slower)

Examples
--------
  .venv/bin/python scripts/herdr_gguf.py --gguf runs/export/lfm2-herdr-Q8_0.gguf \
      --query "split my pane" --spawn
  .venv/bin/python scripts/herdr_gguf.py --gguf run.gguf --query "list tabs in w1" \
      --server-url http://127.0.0.1:8080
  .venv/bin/python scripts/herdr_gguf.py --gguf run.gguf --query "close w1:p4" --cli

Server endpoint (tools=) is documented in the README; this script uses the
pre-rendered prompt so it matches the fine-tune's input format exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from herdr_tools import SCHEMAS  # noqa: E402
import eval_lfm2 as E  # noqa: E402  parse_calls / normalize_call
from transformers import AutoTokenizer  # noqa: E402

MODEL_ID = "LiquidAI/LFM2-350M"
DEFAULT_ENV = "HERDR_ENV=1\nworkspace=w1\ntab=w1:t1\npane=w1:p1\ncwd=/home/repo\nagent kind=hermes"


def render(tok, query: str, system: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": query}],
        tools=SCHEMAS, tokenize=False, add_generation_prompt=True)


class Server:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def generate(self, prompt: str, max_new: int = 128, timeout: int = 120) -> str:
        body = json.dumps({"prompt": prompt, "n_predict": max_new, "temperature": 0,
                           "repeat_penalty": 1.05, "min_p": 0.15, "cache_prompt": True}).encode()
        req = urllib.request.Request(self.url + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)["content"]


def spawn_server(gguf: str, port: int, llama_dir: str) -> subprocess.Popen:
    bin_ = os.path.join(llama_dir, "build", "bin", "llama-server")
    if not os.path.exists(bin_):
        sys.exit(f"llama-server not found at {bin_}")
    proc = subprocess.Popen([bin_, "-m", gguf, "--port", str(port),
                             "--threads", str(os.cpu_count())],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    sys.exit("llama-server did not become healthy")


def cli_generate(prompt: str, gguf: str, max_new: int, llama_dir: str) -> str:
    bin_ = os.path.join(llama_dir, "build", "bin", "llama-cli")
    if not os.path.exists(bin_):
        sys.exit(f"llama-cli not found at {bin_}")
    p = subprocess.run([bin_, "-m", gguf, "-p", prompt, "-st", "-n", str(max_new),
                        "--temp", "0", "--repeat-penalty", "1.05", "--no-display-prompt",
                        "-t", str(os.cpu_count())], capture_output=True, text=True, timeout=600)
    return p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True, help="path to the .gguf")
    ap.add_argument("--query", required=True, help="natural-language Herdr request")
    ap.add_argument("--system", default=DEFAULT_ENV, help="system env block (default: training env)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR", "/tmp/llama.cpp"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--server-url", help="use a running llama-server")
    g.add_argument("--spawn", action="store_true", help="start+stop a temp llama-server")
    g.add_argument("--cli", action="store_true", help="use llama-cli single-turn")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = render(tok, args.query, args.system)
    print("=== prompt (system+user+tools, trimmed) ===")
    print(prompt[:400] + "...\n")

    spawned = None
    if args.server_url:
        gen = Server(args.server_url).generate
    elif args.spawn:
        spawned = spawn_server(args.gguf, args.port, args.llama_cpp_dir)
        gen = Server(f"http://127.0.0.1:{args.port}").generate
    else:
        gen = lambda p, n: cli_generate(p, args.gguf, n, args.llama_cpp_dir)

    try:
        text = gen(prompt, args.max_new_tokens)
    finally:
        if spawned is not None:
            spawned.terminate()
            spawned.wait(timeout=10)

    print("=== raw answer ===")
    print(text)
    print("\n=== parsed tool call(s) ===")
    calls = [E.normalize_call(c) for c in E.parse_calls(text)]
    if not calls:
        print("(no tool call — off-topic refusal)")
    for c in calls:
        print(json.dumps(c, indent=2))


if __name__ == "__main__":
    main()
