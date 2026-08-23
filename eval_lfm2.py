#!/usr/bin/env python3
"""eval_lfm2.py — score the LoRA-tuned LFM2-350M Herdr expert on the same
holdout split as eval_model.py, so numbers are directly comparable.

Planner mode: one turn, greedy, parse tool calls from the native LFM2
[tool_call(...)] syntax in the generated text.
"""
import argparse
import json
import random
import re
from collections import Counter

DATA = "data.jsonl"
MODEL_ID = "LiquidAI/LFM2-350M"


def norm_calls(calls):
    return Counter(
        (c.get("name"), json.dumps(c.get("arguments") or {}, sort_keys=True))
        for c in calls)


def names_of(calls):
    return Counter(c.get("name") for c in calls)


def arg_key(name, calls):
    return Counter(
        json.dumps(c.get("arguments") or {}, sort_keys=True)
        for c in calls if c.get("name") == name)


def parse_calls(text):
    """Parse native LFM2 tool-call syntax: [name(k=v, ...) ...] inside
    <|tool_call_start|>...<|tool_call_end|> (or bare [name(...)] lines)."""
    m = re.search(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", text, re.S)
    body = m.group(1) if m else text
    calls = []
    for name, argstr in re.findall(r"([a-z_][a-z0-9_]*)\((.*?)\)", body, re.S):
        args = {}
        # k=json-value pairs
        for km in re.finditer(
                r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\"(?:[^\"\\]|\\.)*\"|\[[^\]]*\]|-?\d+(?:\.\d+)?|true|false|null)",
                argstr):
            k, v = km.group(1), km.group(2)
            try:
                args[k] = json.loads(v)
            except json.JSONDecodeError:
                args[k] = v
        calls.append({"name": name, "arguments": args})
    return calls


def normalize_call(call: dict) -> dict:
    """Same invariant as ask_herdr.py: pane_split without pane/current
    targets the caller's pane."""
    if call.get("name") == "pane_split":
        args = dict(call.get("arguments") or {})
        if not args.get("pane") and "current" not in args:
            args["current"] = True
        return {"name": call["name"], "arguments": args}
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="lfm2_herdr_lora")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    idx = list(range(len(rows)))
    if args.split > 0:
        random.Random(args.seed).shuffle(idx)
        n_eval = max(1, int(len(rows) * args.split))
        eval_idx = sorted(idx[-n_eval:])
    else:
        eval_idx = list(range(len(rows)))
    eval_run = eval_idx[: args.limit] if args.limit else eval_idx
    print(f"data rows: {len(rows)}   eval rows: {len(eval_run)}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    if not args.base:
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"adapter: {args.adapter}")
    else:
        print("model: BASE (untuned)")
    model.eval()

    def ask(row):
        msgs = []
        if row.get("system"):
            msgs.append({"role": "system", "content": row["system"]})
        msgs.append({"role": "user", "content": row["query"]})
        prompt = tok.apply_chat_template(
            msgs, tools=row.get("tools") or [], tokenize=False,
            add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.05, min_p=0.15)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    exact = exact_norm = tool_ok = n_expected = n_offtopic = off_ok = 0
    arg_counts = {}
    detail = []
    for i in eval_run:
        row = rows[i]
        expected = row["answers"]
        try:
            text = ask(row)
        except Exception as exc:  # noqa: BLE001
            detail.append((i, row["query"], expected, [], f"ERROR: {exc!r}"))
            continue
        predicted = parse_calls(text)
        normalized = [normalize_call(c) for c in predicted]
        pn, en = norm_calls(predicted), norm_calls(expected)
        pn_norm = norm_calls(normalized)
        pw, ew = names_of(predicted), names_of(expected)
        if en:
            n_expected += 1
            exact += (pn == en)
            exact_norm += (pn_norm == en)
            tool_ok += (pw == ew)
            for name in ew:
                ok = arg_key(name, predicted) == arg_key(name, expected)
                rec = arg_counts.setdefault(name, [0, 0])
                rec[0] += ok
                rec[1] += 1
        else:
            n_offtopic += 1
            off_ok += (not predicted)
            if predicted:
                detail.append((i, row["query"], [], predicted,
                               "OFF-TOPIC but called a tool"))
                continue
        if en and pn != en:
            detail.append((i, row["query"], expected, predicted, ""))

    print("== summary ==")
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
        print("\n== mismatches (first 10) ==")
        for i, q, exp, pred, note in detail[:10]:
            print(f"  row {i}: {q[:80]}")
            if note:
                print(f"    {note}")
            print(f"    expected : {json.dumps(exp)[:150]}")
            print(f"    predicted: {json.dumps(pred)[:150]}")


if __name__ == "__main__":
    main()
