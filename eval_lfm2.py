#!/usr/bin/env python3
"""eval_lfm2.py — score the LoRA-tuned LFM2-350M Herdr expert on the same
holdout split as eval_model.py, so numbers are directly comparable.

Planner mode: one turn, greedy, parse tool calls from the native LFM2
[tool_call(...)] syntax in the generated text.
"""
import argparse
import ast
import json
import re
from collections import Counter

from split import eval_holdout, load_pinned_holdout

DATA = "dataset.jsonl"
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


def _split_top_level(s, sep=","):
    """Split s on sep, ignoring separators inside (), [], {} or quotes."""
    parts, cur, depth, quote = [], [], 0, None
    i = 0
    while i < len(s):
        c = s[i]
        if quote:
            cur.append(c)
            if c == "\\" and i + 1 < len(s):
                cur.append(s[i + 1])
                i += 1
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
            cur.append(c)
        elif c in "([{":
            depth += 1
            cur.append(c)
        elif c in ")]}":
            depth -= 1
            cur.append(c)
        elif c == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _parse_value(v):
    """Parse one rendered arg value: 'str', \"json\", True/False/None,
    number, [list], {dict}."""
    v = v.strip()
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v in ("null", "None"):
        return None
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        if v[0] == "'":
            # Single-quoted (chat-template format_arg_value) — literal_eval
            # unescapes \\, \', \n the same way the template escaped them.
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return v[1:-1]
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v[1:-1]
    if v[:1] in ("[", "{"):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _find_bracket_blocks(text):
    """Yield contents of top-level [...] blocks, respecting nesting and
    quotes (so a list arg like [\"done\"] doesn't end the block early)."""
    i, n = 0, len(text)
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        depth, quote = 0, None
        j = i + 1
        while j < n:
            c = text[j]
            if quote:
                if c == "\\":
                    j += 1
                elif c == quote:
                    quote = None
            elif c in "'\"":
                quote = c
            elif c == "[":
                depth += 1
            elif c == "]":
                if depth == 0:
                    yield text[i + 1:j]
                    i = j
                    break
                depth -= 1
            j += 1
        i += 1


def _parse_call_chunks(body):
    """Parse a comma-separated call list (no outer brackets) into calls."""
    calls = []
    for chunk in _split_top_level(body, ","):
        mm = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", chunk)
        if not mm:
            continue
        name = mm.group(1)
        j = mm.end()
        depth, quote, k = 1, None, j
        while k < len(chunk) and depth > 0:
            c = chunk[k]
            if quote:
                if c == "\\":
                    k += 1
                elif c == quote:
                    quote = None
            elif c in "'\"":
                quote = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            k += 1
        argstr = chunk[j:k - 1]
        args = {}
        for part in _split_top_level(argstr, ","):
            if "=" in part:
                key, _, val = part.partition("=")
                args[key.strip()] = _parse_value(val)
        calls.append({"name": name, "arguments": args})
    return calls


def parse_calls(text):
    """Parse LFM2 tool calls from generated text.

    Primary format: native
      reasoning<|tool_call_start|>[name(k=v, ...), ...]<|tool_call_end|>
    Fallback (base model / old adapters): bare [name(k=v, ...)] blocks.
    """
    m = re.search(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", text, re.S)
    if m:
        body = m.group(1).strip()
        if body.startswith("[") and body.endswith("]"):
            body = body[1:-1]
        return _parse_call_chunks(body)
    calls = []
    for block in _find_bracket_blocks(text):
        calls.extend(_parse_call_chunks(block))
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
    ap.add_argument("--adapter", default="adapters/lfm2_herdr_lora")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", default=None,
                    help="pinned eval holdout JSON (keyed by query); overrides --split/--seed")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.holdout:
        eval_idx, hmeta = load_pinned_holdout(args.holdout, rows)
        print(f"holdout: {args.holdout} (pinned {len(eval_idx)} rows, n={hmeta['n']})")
    elif args.split > 0:
        eval_idx = eval_holdout(len(rows), args.split, args.seed)
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
        # Prompt with system+user ONLY; the assistant turn is what we score.
        # (Using row["messages"] would include the gold answer and ask the
        # model to continue — train_lfm2.py masks with messages[:-1], so eval
        # must match.)
        prompt = tok.apply_chat_template(
            row["messages"][:-1], tools=row.get("tools") or [],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.05, min_p=0.15)
        # Keep special tokens: skip_special_tokens=True would strip ids 10/11
        # (<|tool_call_start|>/<|tool_call_end|>), which parse_calls needs for
        # native-format boundaries.
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)

    exact = exact_norm = tool_ok = n_expected = n_offtopic = off_ok = 0
    arg_counts = {}
    detail = []
    for i in eval_run:
        row = rows[i]
        expected = row["expected"]
        try:
            text = ask(row)
        except Exception as exc:  # noqa: BLE001
            detail.append((i, row["messages"][1]["content"], expected, [], f"ERROR: {exc!r}"))
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
                detail.append((i, row["messages"][1]["content"], [], predicted,
                               "OFF-TOPIC but called a tool"))
                continue
        if en and pn != en:
            detail.append((i, row["messages"][1]["content"], expected, predicted, ""))

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
