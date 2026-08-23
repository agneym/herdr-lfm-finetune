#!/usr/bin/env python3
"""eval_model.py — score a tuned (or base) Needle 2 Herdr expert against labels.

The primary eval: run every query through the agent in PLANNER mode (no side
effects), compare the emitted tool calls to the ground-truth `answers` in the
dataset, and report:

  - exact-call accuracy   (name + every argument matches, order-insensitive)
  - tool-selection accuracy (just the tool names, as a multiset)
  - argument-grounding accuracy (per tool: instances with exact args)
  - off-topic behaviour   (rows with answers:[] must emit no tool call)

Usage
    # tuned model, held-out slice (train on the complement rows!)
    .venv/bin/python eval_model.py --weights tuned.cact --split 0.15

    # whole-set smoke test (optimistic — these rows were seen in training)
    .venv/bin/python eval_model.py --weights tuned.cact --split 0

    # baseline: the UNTUNED base model, same queries (must run in its own
    # process — the Needle engine holds one set of weights per process)
    .venv/bin/python eval_model.py --base --split 0.15

    # optional: dump predictions for a live herdr replay pass
    .venv/bin/python eval_model.py --weights tuned.cact --save-preds preds.jsonl

The `--split` slice is deterministic (fixed seed) and its row indices are
printed, so you can hold exactly those rows out of training.
"""
import argparse
import json
import random
import sys
from collections import Counter

import herdr_tools as ht
from ask_herdr import build_system

DATA = "data.jsonl"


def norm_calls(calls):
    """Counter of (name, canonical-args) — order-insensitive multiset.

    Args are canonicalised with json.dumps(sort_keys=True) so nested
    lists (e.g. pane_send_keys `keys`) stay hashable.
    """
    return Counter(
        (c.get("name"), json.dumps(c.get("arguments") or {}, sort_keys=True))
        for c in calls
    )


def names_of(calls):
    return Counter(c.get("name") for c in calls)


def arg_key(name, calls):
    """Counter of canonical-args for one tool name (per-name grounding check)."""
    return Counter(
        json.dumps(c.get("arguments") or {}, sort_keys=True)
        for c in calls if c.get("name") == name
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval the Herdr Needle expert.")
    ap.add_argument("--weights", default=None, help="tuned .cact (default: none)")
    ap.add_argument("--base", action="store_true",
                    help="eval the untuned base model instead (no .cact)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--split", type=float, default=0.15,
                    help="fraction held out for eval (0 = whole set)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="max eval queries (0 = all) — for quick smoke runs")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--save-preds", default=None,
                    help="write per-query predictions as JSONL (for live replay)")
    ap.add_argument("--save-train", default=None,
                    help="write data.jsonl minus the eval rows here, so the "
                         "holdout is truly unseen during training")
    args = ap.parse_args()

    if args.base and args.weights:
        sys.exit("pick one: --base OR --weights, not both")

    rows = [json.loads(l) for l in open(args.data)]
    idx = list(range(len(rows)))
    if args.split > 0:
        random.Random(args.seed).shuffle(idx)
        n_eval = max(1, int(len(rows) * args.split))
        eval_idx = sorted(idx[-n_eval:])
    else:
        eval_idx = list(range(len(rows)))
    # --limit only shrinks the EVAL run; the full holdout is still excluded
    # from any --save-train output so the split stays clean.
    eval_run = eval_idx[: args.limit] if args.limit else eval_idx
    eval_set = set(eval_idx)
    if args.save_train:
        with open(args.save_train, "w") as f:
            for i, row in enumerate(rows):
                if i not in eval_set:
                    f.write(json.dumps(row) + "\n")
        print(f"wrote training set ({len(rows) - len(eval_set)} rows) -> "
              f"{args.save_train}  (eval rows excluded)")
    print(f"data rows: {len(rows)}   eval rows: {len(eval_run)}"
          f"   (split {args.split}, seed {args.seed})")
    print("eval row indices:", eval_run)
    print("NOTE: rows above are the holdout — train on the complement.")

    import needle

    weights = None if args.base else args.weights
    agent = needle.Needle(tools=ht.get_tools(), system=build_system(),
                          weights=weights)
    print(f"model: {'BASE (untuned)' if args.base else weights}\n")

    preds_out = open(args.save_preds, "w") if args.save_preds else None
    detail = []          # (idx, query, expected, predicted, note)
    exact = tool_ok = n_expected = n_offtopic = off_ok = 0
    arg_counts = {}          # tool -> [instances ok, instances expected]
    err_rows = 0

    for i in eval_run:
        row = rows[i]
        expected = row["answers"]
        try:
            resp = agent.complete(row["query"], max_new_tokens=args.max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            err_rows += 1
            detail.append((i, row["query"], expected, [], f"ERROR: {exc!r}"))
            continue
        predicted = resp.get("function_calls") or []
        if resp.get("type") == "text" and not predicted:
            predicted = []  # off-topic answer; no call
        if preds_out is not None:
            preds_out.write(json.dumps(
                {"row": i, "query": row["query"], "expected": expected,
                 "predicted": predicted, "type": resp.get("type")}) + "\n")

        pn, en = norm_calls(predicted), norm_calls(expected)
        pw, ew = names_of(predicted), names_of(expected)
        if en:
            n_expected += 1
            exact += (pn == en)
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

    if preds_out is not None:
        preds_out.close()

    n = len(eval_run) - err_rows
    print("== summary ==")
    print(f"  eval rows            : {n}  (errors: {err_rows})")
    if n:
        print(f"  exact-call accuracy  : {exact}/{n_expected} = "
              f"{100*exact/n_expected:.1f}%   (of {n_expected} tool rows)")
        print(f"  tool-selection acc   : {tool_ok}/{n_expected} = "
              f"{100*tool_ok/n_expected:.1f}%")
        if n_offtopic:
            print(f"  off-topic (no call)  : {off_ok}/{n_offtopic} = "
                  f"{100*off_ok/n_offtopic:.1f}%   (of {n_offtopic} off-topic rows)")

    print("\n== per-tool argument grounding ==")
    for name in sorted(arg_counts):
        ok, total = arg_counts[name]
        bad = total - ok
        print(f"  {name:<22} {ok}/{total} exact "
              f"({100*ok/total:.0f}%)"
              + (f"  <-- {bad} instance(s) with wrong args" if bad else ""))

    if detail:
        print("\n== mismatches (first 10) ==")
        for i, q, exp, pred, note in detail[:10]:
            print(f"  row {i}: {q[:80]}")
            if note:
                print(f"    {note}")
            print(f"    expected : {json.dumps(exp)[:150]}")
            print(f"    predicted: {json.dumps(pred)[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
