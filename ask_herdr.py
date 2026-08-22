#!/usr/bin/env python3
"""ask_herdr.py — query the fine-tuned Herdr Needle 2 model.

The main agent (Hermes) calls this to turn a natural-language Herdr request into
the exact herdr operation(s).  By default it runs the model in PLANNER mode
(one turn, no side effects): it returns the tool call(s) the model chooses plus a
ready-to-run `herdr ...` command for Hermes to execute.

    python ask_herdr.py --query "split my pane to the right and run the linter"
    echo "read the last 120 lines of pane w1:p1" | python ask_herdr.py

Options
    --query TEXT      the natural-language request (or pass via stdin)
    --weights PATH    .cact to load (default tuned.cact)
    --run             self-execute the chosen operations via the tool functions
                      (set NEEDLE_HERDR_EXECUTE=1 to actually shell out to herdr)
    --max-new-tokens int   (default 256)
"""
import argparse
import json
import os
import sys

import herdr_tools as ht


def build_system() -> str:
    """Assemble environment facts the model grounds 'current pane' / cwd against."""
    def env(key, fallback):
        return os.environ.get(key) or fallback
    return (
        f"HERDR_ENV={env('HERDR_ENV', '1')}\n"
        f"workspace={env('HERDR_WORKSPACE_ID', 'w1')}\n"
        f"tab={env('HERDR_TAB_ID', 'w1:t1')}\n"
        f"pane={env('HERDR_PANE_ID', 'w1:p1')}\n"
        f"cwd={env('PWD', '/home/repo')}\n"
        f"agent kind={env('HERDR_AGENT_KIND', 'hermes')}"
    )


def render_operation(call: dict) -> dict:
    """Map a model call {name, arguments} back to a herdr command using the
    same command-builder the tool functions use (EXECUTE off => no side effect)."""
    name = call.get("name")
    args = call.get("arguments") or {}
    fn = getattr(ht, name, None)
    if fn is None:
        return {"name": name, "error": "unknown tool", "arguments": args}
    try:
        desc = fn(**args)  # returns {"name","command","arguments"}
        return {"name": name, "command": desc["command"], "arguments": args}
    except TypeError as exc:
        return {"name": name, "error": f"bad arguments: {exc}", "arguments": args}


def main() -> int:
    ap = argparse.ArgumentParser(description="Query the tuned Herdr Needle model.")
    ap.add_argument("--query", help="natural-language Herdr request")
    ap.add_argument("--weights", default="tuned.cact", help="path to tuned .cact")
    ap.add_argument("--run", action="store_true",
                    help="self-execute chosen operations (loop) instead of planning")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    query = args.query or sys.stdin.read().strip() if not sys.stdin.isatty() else args.query
    if not query:
        print("no query provided (use --query or pipe via stdin)", file=sys.stderr)
        return 2

    import needle

    weights = args.weights
    if not os.path.exists(weights):
        print(f"warning: weights '{weights}' not found; running the base model "
              f"(build a tuned .cact to get the Herdr expert).", file=sys.stderr)
        weights = None

    agent = needle.Needle(
        tools=ht.get_tools(),
        system=build_system(),
        weights=weights,
    )

    if args.run:
        resp = agent.run(query, max_new_tokens=args.max_new_tokens)
        print(json.dumps(resp, indent=2, default=str))
        return 0

    resp = agent.complete(query, max_new_tokens=args.max_new_tokens)
    rtype = resp.get("type")
    calls = resp.get("function_calls") or []

    if rtype != "call" or not calls:
        # Plain-text answer (e.g. off-topic or a refusal).
        out = {"type": rtype, "text": resp.get("text") or resp.get("response", "")}
        print(json.dumps(out, indent=2))
        return 0

    operations = [render_operation(call) for call in calls]
    print(json.dumps({"type": "call", "operations": operations}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
