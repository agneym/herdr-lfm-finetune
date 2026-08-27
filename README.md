# Herdr expert: LFM2-350M LoRA

Fine-tune **LiquidAI/LFM2-350M** (PEFT LoRA) to be an expert on the **Herdr**
terminal multiplexer, so a main agent can ask it in natural language for the
right Herdr operation and run the result.

> An earlier attempt used Needle 2 (cactus-needle). It produced degenerate
> output (duplicated keys, garbled args) from its trainer and was abandoned —
> see `NOTES.md` for the numbers and the commit to go back to.

---

## How it works

The pipeline is three steps:

1. `make_dataset.py` generates `dataset.jsonl` — ~656 examples of
   `{messages, tools, expected}` covering all 25 Herdr ops, ~12% off-topic
   including hard negatives that reuse Herdr verbs but act outside Herdr.
   Tail tools get >=10 surface forms; contrastive minimal pairs separate
   confusable ops (list/create worktrees, get/install).
   The tool schemas come from `herdr_tools.py`. `messages` are chat-format,
   ready for `tokenizer.apply_chat_template(tools=...)`: on-topic assistant
   turns keep the reasoning line and carry a structured `tool_calls` field,
   which the chat template renders in native
   `<|tool_call_start|>[name(k=v, ...)]<|tool_call_end|>` syntax; off-topic
   rows learn a natural-language refusal. `expected` carries the structured
   tool-call labels for validate/eval.
2. `train_lfm2.py` — PEFT LoRA SFT (loss masked to assistant tokens only),
   saves the best-validation checkpoint.
3. `eval_lfm2.py` — scores on a deterministic holdout (seed 42, last 15%),
   reporting raw AND normalized exact-call accuracy.

## Files

| File | Purpose |
|------|---------|
| `herdr_tools.py` | The 25 Herdr operations; schemas loaded from `reference/herdr_schemas.json`. |
| `make_dataset.py` | Generates `dataset.jsonl` (chat format + structured labels). |
| `train_lfm2.py` | LoRA SFT driver (run on Colab GPU; see below). |
| `eval_lfm2.py` | Holdout eval; `--base` for baseline. |
| `pin_holdout.py` | Persists the eval holdout (keyed by query) so re-eval stays comparable as the dataset grows. |
| `validate_dataset.py` | Live-validates dataset labels against a real `herdr` server. |
| `adapters/lfm2_herdr_lora/` | Current tuned adapter. `lfm2_herdr_lora_v1/` is the previous run. |
| `reference/` | Herdr tool schemas, captured CLI help (`cli_help/`), API schema, skill doc. |
| `scripts/` | Colab glue (`setup_lfm2_colab.py`, `fix_torchao.py`, `run_detached_*.py`) and one-off probes. |
| `runs/` | Experiment artifacts (gitignored): checkpoint tarballs, training logs, eval snapshots. |
| `NOTES.md` | Why we dropped Needle 2; how to recover that track. |
| `Makefile` | `make data` / `make eval` / `make validate`; `make train` prints the Colab recipe. |

The pipeline chain is: `make data` -> train on Colab -> unpack the dumped
checkpoint into `adapters/lfm2_herdr_lora/` -> `make eval`.

## Train on Google Colab

A T4 is enough for 350M (~15 min). Use the `colab` CLI (or just run
`make train` to print this recipe):

```sh
colab new -s NAME --gpu T4
colab exec -s NAME -f scripts/setup_lfm2_colab.py   # transformers>=4.55 peft datasets accelerate
colab exec -s NAME --timeout 400 -f scripts/fix_torchao.py   # torchao>=0.16 (peft 0.20 requires it)
colab upload -s NAME dataset.jsonl /content/dataset.jsonl
colab upload -s NAME train_lfm2.py /content/train_lfm2.py
colab upload -s NAME split.py /content/split.py     # train_lfm2.py imports it
colab upload -s NAME runs/eval_v5_holdout.json /content/eval_v5_holdout.json   # optional: pinned holdout
```

Optional but recommended — mirror the checkpoint to Google Drive so it
survives a VM reap: `colab drivemount -s NAME` (interactive OAuth once), then
queue `scripts/copy_to_drive.py` as a second `colab exec` (it waits for the
checkpoint tarball and copies it to `/content/drive/MyDrive/herdr/`, sha256
verified).

(The Colab scripts write the adapter flat at `/content/lfm2_herdr_lora`; after
reconstructing the dumped tarball locally, unpack it into `adapters/`.)

NEVER run training inside one blocking `colab exec` — an exec timeout or a dead
keep-alive daemon lets Colab idle-prune the VM and you lose the checkpoint.
Use `run_detached_dump.py`: nohup-detach the trainer, poll every 120 s, tick
keep-alive every 60 s, then tar+base64-dump the checkpoint to stdout on
completion (the VM can be reaped seconds after training finishes). Reconstruct
locally with the snippet in `NOTES.md` / the script's docstring. To train with
the pinned holdout (so the 98 eval rows never leak into training), pass
`--env HOLDOUT=/content/eval_v5_holdout.json` to that `colab exec`.

Hyperparameters that worked (v3, 314 rows): epochs 8, batch 1, grad-accum 8,
lr 1e-4, LoRA r=16 alpha=32 on q/k/v_proj ONLY (do NOT target conv
in_proj/out_proj — PEFT routes them through torchao and crashes).

v4 recipe (656 rows): epochs 12 (v3 val loss still improving at 8), same
r/alpha/lr, targets q/k/v_proj + w1/w3/w2. LFM2 naming gotcha: the MLP
projections are `w1`/`w3`/`w2`, NOT gate/up/down_proj (those match nothing),
and attention output is `out_proj` which is SHARED with Lfm2ShortConv —
there is no `o_proj` at all, so the old "o_proj" target silently trained
q/k/v only.

## Evaluate

```sh
.venv/bin/python eval_lfm2.py --adapter adapters/lfm2_herdr_lora --split 0.15
.venv/bin/python eval_lfm2.py --base          # baseline
```

To keep numbers comparable as the dataset grows, pin the holdout once (keyed
by query string, so appending training rows never shifts it) and reuse it with
`--holdout` on both eval and train:

```sh
.venv/bin/python pin_holdout.py --data dataset.jsonl --out runs/eval_v5_holdout.json
.venv/bin/python eval_lfm2.py --adapter adapters/lfm2_herdr_lora --holdout runs/eval_v5_holdout.json
```

Honest numbers (98-row holdout, seed 42, strictly disjoint from training,
all 25 tools represented) — after fixing an eval bug that had been feeding the
gold answer back into the prompt for every prior run:

| model | exact-call | exact-norm | tool-selection | off-topic |
|---|---:|---:|---:|---:|
| base (untuned)              | 9.8%  | 9.8%  | 26.8% | 68.8% |
| v4 (bare format)            | 70.7% | 75.6% | 91.5% | 100.0% |
| v5 (native format)          | 85.4% | 85.4% | 95.1% | 100.0% |
| **v6 (targeted fixes)**     | **93.9%** | **93.9%** | **96.3%** | **100.0%** |

See `runs/eval_v6_summary.md` (and `runs/eval_v6_new.log` /
`eval_v5_new.log` / `eval_v4_honest.log` / `eval_base_honest.log`). v6 keeps
v5's native `<|tool_call_start|>[name(k=v)]<|tool_call_end|>` syntax and adds
append-only rows targeting the v5 holdout failures; the eval holdout is pinned
(`runs/eval_v5_holdout.json`) so all versions score the same 98 rows.

> All previously-published numbers (v1-v4, e.g. 50.0% / 65.7%) are invalid:
> eval's `ask()` rendered `row["messages"]` (including the gold assistant
> answer) with `add_generation_prompt=True`, scoring a *continuation* rather
> than a from-scratch answer. Fixed to `row["messages"][:-1]` to match
> `train_lfm2.py`. Re-run any adapter with the fixed `eval_lfm2.py` before
> comparing against the table above.

Runtime invariant: `pane_split` without explicit pane/current targets the
caller's pane; normalization makes it explicit (`current: true`). Eval reports
both raw and normalized accuracy.

## Safety notes

- Eval is side-effect free; it only compares predicted calls to labels.
- Do not close workspaces/tabs/panes/sessions you did not create, and never
  `herdr server stop` from an active session unless intended.
- Command syntax follows the installed `herdr` CLI (v0.8.2), captured under
  `cli_help/` — the installed CLI is the source of truth.
