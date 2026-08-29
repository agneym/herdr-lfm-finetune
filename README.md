# Herdr expert: LFM2-350M LoRA

Fine-tune **LiquidAI/LFM2-350M** (PEFT LoRA) to be an expert on the **Herdr**
terminal multiplexer, so a main agent can ask it in natural language for the
right Herdr operation and run the result.

![Herdr holdout accuracy: base vs fine-tune vs frontier models](docs/eval_comparison.png)

*A 350M LoRA fine-tune turns a 9.8% base model into a 96.3% expert — beating
deepseek flash (56.1%) and GLM 5.3 flash (73.2%) on Herdr tool-calling, on the
pinned 98-row holdout.*

**Current state:** the tuned adapter is `adapters/lfm2_herdr_lora` (the "v7"
run). Current dataset is 804 rows (98 marked off-topic — 12.2%) across all 25
Herdr ops. See `runs/eval_v7_summary.md` for the run history and per-version
comparisons.

---

## Quick start

Load the tuned LoRA adapter and ask one prompt in ~30 seconds:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from herdr_tools import SCHEMAS                      # the 25 Herdr tool schemas

tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")
model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2-350M", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "adapters/lfm2_herdr_lora").eval()

prompt = tok.apply_chat_template(
    [{"role": "system", "content":
        "HERDR_ENV=1\nworkspace=w1\ntab=w1:t1\npane=w1:p1\ncwd=/home/repo\nagent kind=hermes"},
     {"role": "user", "content": "split my pane"}],
    tools=SCHEMAS, tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=192, do_sample=False)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=False))
```

The model answers in native `<|tool_call_start|>[name(k=v, ...)]<|tool_call_end|>`
syntax; parse it with `eval_lfm2.parse_calls`. The full planner loop and the
98-row holdout eval live in `eval_lfm2.py` (see [Evaluate](#evaluate)). Training
runs on a Google Colab GPU — see [Training](#training-google-colab).

---

## How it works

The pipeline is three steps:

1. `make_dataset.py` generates `dataset.jsonl` — `{messages, tools, expected}`
   entries covering all 25 Herdr ops, ~12% off-topic including hard negatives
   that reuse Herdr verbs but act outside Herdr. Tail tools get >=10 surface
   forms; contrastive minimal pairs separate confusable ops (list/create
   worktrees, get/install). The tool schemas come from `herdr_tools.py`.
   `messages` are chat-format, ready for
   `tokenizer.apply_chat_template(tools=...)`: on-topic assistant turns keep
   the reasoning line and carry a structured `tool_calls` field, which the
   chat template renders in native
   `<|tool_call_start|>[name(k=v, ...)]<|tool_call_end|>` syntax; off-topic
   rows learn a natural-language refusal. `expected` carries the structured
   tool-call labels for validate/eval.
2. `train_lfm2.py` — PEFT LoRA SFT (loss masked to assistant tokens only),
   saves the best-validation checkpoint (training runs on a Colab GPU — see below).
3. `eval_lfm2.py` — scores against a pinned, query-keyed holdout, reporting raw
   AND normalized exact-call accuracy.

The deterministic split logic lives in `split.py`, the single source of truth:
the eval holdout is carved out first, then train/val are split from the rest,
so the three sets are provably disjoint.

## Files

| File | Purpose |
|------|---------|
| `herdr_tools.py` | The 25 Herdr operations; schemas loaded from `reference/herdr_schemas.json`. |
| `split.py` | Single source of truth for the train/val/eval split (holdout carved out first). |
| `make_dataset.py` | Generates `dataset.jsonl` (chat format + structured labels). |
| `train_lfm2.py` | LoRA SFT driver (run on Colab GPU; see below). |
| `eval_lfm2.py` | Holdout eval; `--base` for baseline. |
| `eval_pi.mjs` | Same holdout, scored through pi's `ModelRuntime` for any catalog model (deepseek flash, GLM 5.3 flash); records tokens + cost. |
| `pin_holdout.py` | Persists the eval holdout (keyed by query) so re-eval stays comparable as the dataset grows. |
| `validate_dataset.py` | Live-validates dataset labels against a real `herdr` server. |
| `adapters/lfm2_herdr_lora/` | **Current tuned adapter (v7).** Older runs are archived alongside (`_v1`, `_v3`, `_v4`, `_v6`). |
| `reference/` | Herdr tool schemas, captured CLI help (`cli_help/`), API schema, skill doc. |
| `scripts/` | Colab glue (`setup_lfm2_colab.py`, `fix_torchao.py`, `run_detached_*.py`), one-off probes (`probe_*`), and `make_eval_graph.py` (renders `docs/eval_comparison.*`). |
| `runs/` | Experiment artifacts and durable results: checkpoint tarballs, training logs, eval snapshots + per-version `eval_v*_summary.md`; [`runs/caveats.md`](runs/caveats.md) is the compatibility/caveats doc. |
| `NOTES.md` | Why we dropped Needle 2; how to recover that track. |
| `Makefile` | `make data` / `make eval` / `make validate`; `make train` prints the Colab recipe. |

The pipeline chain is: `make data` -> train on Colab -> unpack the dumped
checkpoint into `adapters/lfm2_herdr_lora/` -> `make eval`.

## Training (Google Colab)

A T4 is enough for 350M (~15 min). Use the `colab` CLI (or just run
`make train` to print this recipe):

```sh
colab new -s NAME --gpu T4
colab exec -s NAME -f scripts/setup_lfm2_colab.py   # transformers>=4.55 peft accelerate (datasets dropped)
colab exec -s NAME --timeout 400 -f scripts/fix_torchao.py   # torchao>=0.16 (peft 0.20 requires it)
colab upload -s NAME dataset.jsonl /content/dataset.jsonl
colab upload -s NAME train_lfm2.py /content/train_lfm2.py
colab upload -s NAME split.py /content/split.py     # train_lfm2.py imports it
colab upload -s NAME runs/eval_v5_holdout.json /content/eval_v5_holdout.json   # optional: pinned holdout
```

Current recipe (v6/v7): epochs 12, batch 1, grad-accum 8, lr 1e-4, LoRA
r=16 alpha=32 on `q/k/v` + `w1/w3/w2`. The Colab scripts write the adapter flat
at `/content/lfm2_herdr_lora`; unpack the dumped tarball into `adapters/` after
reconstructing it locally. Train with
`--env HOLDOUT=/content/eval_v5_holdout.json` so the 98 eval rows never leak
into training. The LFM2 target-map gotcha, the Drive-mirror / VM-reap lifecycle,
and the exact `run_detached_dump.py` flow are in
[`runs/caveats.md`](runs/caveats.md).

## Evaluate

Score the adapter against the pinned, query-keyed holdout (98 rows) so results
stay comparable as the dataset grows:

```sh
.venv/bin/python eval_lfm2.py --adapter adapters/lfm2_herdr_lora --holdout runs/eval_v5_holdout.json
.venv/bin/python eval_lfm2.py --base --holdout runs/eval_v5_holdout.json   # baseline
```

The holdout is pinned once via `pin_holdout.py` (keyed by query string, so
appending training rows never shifts it) and reused for both eval and train —
how the pin and split stay valid is in [`runs/caveats.md`](runs/caveats.md).

**Current result (v7)** — 98-row holdout, seed 42, strictly disjoint from
training, all 25 tools represented:

| model | exact-call | exact-norm | tool-selection | off-topic |
|---|---:|---:|---:|---:|
| base (untuned) | 9.8% | 9.8% | 26.8% | 68.8% |
| **v7 (current adapter)** | **96.3% (79/82)** | **96.3%** | **96.3%** | **100% (16/16)** |
| deepseek-v4-flash-vision-exp (pi harness) | 56.1% | 56.1% | 65.9% | 50.0% |
| glm-5.3-flash / OpenRouter (pi harness) | 73.2% | 76.8% | 87.8% | 50.0% |

> **Frontier-model comparison** (`eval_pi.mjs`): deepseek-v4-flash-vision-exp
> and glm-5.3-flash are scored through pi's `ModelRuntime` with the same 25
> Herdr tools and the same pinned holdout. Both trail the fine-tune badly on
> exact-call (56.1% and 73.2% vs 96.3%) and both act on half the off-topic
> prompts; the 98-row runs cost **$0.0069** (deepseek) and **$0.0080** (GLM).
> Full breakdowns: `runs/eval_deepseek_summary.md`, `runs/eval_glm_summary.md`.

> **Before comparing runs, read [`runs/caveats.md`](runs/caveats.md).** v7 is a
> *regime change* (rotated system prompts, context-free grounding) and must not
> be compared head-to-head with v6; every pre-fix number (v1–v4, before commit
> `798b7d8`) scored a *continuation* and is invalid. Per-run numbers and failure
> modes are in `runs/eval_v7_summary.md`.

Runtime invariant: `pane_split` without explicit pane/current targets the
caller's pane; normalization makes it explicit (`current: true`). Eval reports
both raw and normalized accuracy.

## Limitations

The honest failure modes on the pinned 98-row holdout (v7), from
`runs/eval_v7_summary.md`:

- **"give me a new pane on the right"** — still emits a hallucinated
  `pane_create(Direction=...)` (wrong casing). "give me a new pane"
  paraphrases are thin in training; not fixed since v6.
- **"where am i?" / "please, where am i?"** — under-calls (emits no tool).
  These surface forms are held out and absent from training; `pane_current` has
  ~20 training forms but not these.
- **Two tools are below 100% grounding** — `pane_split` 4/5 and `pane_current`
  1/3 on exact args; the other 16 tools ground at 100%.

Beyond the holdout, `validate_dataset.py` replays against a live `herdr` server
and reports **23 FAIL** (22× `integration_install` for agent kinds the CLI
doesn't recognize yet — cursor/copilot/devin/droid/kilo — plus 1×
`worktree_create` with `base='develop'`).

This is a **narrow specialist**, not a general model: it plans the 25 Herdr ops
and refuses off-topic prompts; it does not do general chat, code, or reasoning.

## Safety notes

- Eval is side-effect free; it only compares predicted calls to labels.
- Do not close workspaces/tabs/panes/sessions you did not create, and never
  `herdr server stop` from an active session unless intended.
- Command syntax follows the installed `herdr` CLI (v0.8.2), captured under
  `cli_help/` — the installed CLI is the source of truth.

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and
redistribute for any purpose, including commercial use and distillation. See
the `LICENSE` file for the full text.
