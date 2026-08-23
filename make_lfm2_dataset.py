#!/usr/bin/env python3
"""make_lfm2_dataset.py — convert herdr data.jsonl into LFM2 chat-format JSONL.

Each output row is {"messages": [...], "tools": [...]} ready for
tokenizer.apply_chat_template(tools=...) during SFT. The assistant turn keeps
the reasoning line (as plain text before the tool call) and emits calls in the
native LFM2 tool-call syntax the template produces.

Rows with answers:[] keep a text-only assistant reply derived from `reasoning`
(or a generic refusal) so off-topic behaviour is learned too.
"""
import json
import sys

DATA = "data.jsonl"
OUT = "data_lfm2.jsonl"


def convert(row):
    tools = row.get("tools") or []
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": row["system"]})
    messages.append({"role": "user", "content": row["query"]})

    answers = row.get("answers") or []
    reasoning = (row.get("reasoning") or "").strip()
    if answers:
        # native LFM2 call syntax: [name(arg="val", ...)]
        calls = ", ".join(
            name + "(" + ", ".join(
                json.dumps(v) if not isinstance(v, str)
                else f"{k}={json.dumps(v)}"
                for k, v in (a.get("arguments") or {}).items()
            ) + ")"
            for a, name in ((a, a["name"]) for a in answers)
        )
        content = f"[{calls}]"
        if reasoning:
            content = reasoning + "\n" + content
        messages.append({"role": "assistant", "content": content})
    else:
        content = ("This request isn't a Herdr terminal operation, so no tool "
                   "call is needed.")
        if reasoning:
            content = reasoning
        messages.append({"role": "assistant", "content": content})
    return {"messages": messages, "tools": tools}


def main():
    n = 0
    with open(DATA) as fin, open(OUT, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps(convert(json.loads(line))) + "\n")
            n += 1
    print(f"wrote {n} rows -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
