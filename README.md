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

1. `make_dataset.py` generates `dataset.jsonl` — 270 examples of
   `{messages, tools, expected}` covering all 25 Herdr ops, ~12% off-topic.
   The tool schemas come from `herdr_tools.py`. `messages` are chat-format,
   ready for `tokenizer.apply_chat_template(tools=...)`: assistant turns keep
   the reasoning line, then emit calls in native LFM2 syntax `[name(arg=val)]`;
   off-topic rows learn a text-only refusal. `expected` carries the structured
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
```

(The Colab scripts write the adapter flat at `/content/lfm2_herdr_lora`; after
reconstructing the dumped tarball locally, unpack it into `adapters/`.)

NEVER run training inside one blocking `colab exec` — an exec timeout or a dead
keep-alive daemon lets Colab idle-prune the VM and you lose the checkpoint.
Use `run_detached_dump.py`: nohup-detach the trainer, poll every 120 s, tick
keep-alive every 60 s, then tar+base64-dump the checkpoint to stdout on
completion (the VM can be reaped seconds after training finishes). Reconstruct
locally with the snippet in `NOTES.md` / the script's docstring.

Hyperparameters that worked (270 rows): epochs 8, batch 1, grad-accum 8,
lr 1e-4, LoRA r=16 alpha=32 on q/k/v/o_proj ONLY (do NOT target conv
in_proj/out_proj — PEFT routes them through torchao and crashes).

## Evaluate

```sh
.venv/bin/python eval_lfm2.py --adapter adapters/lfm2_herdr_lora --split 0.15
.venv/bin/python eval_lfm2.py --base          # baseline
```

Current numbers (47-row holdout, seed 42, strictly disjoint from training):
exact-call **31.7%** / exact-norm **46.3%**, tool-selection **63.4%**,
off-topic restraint **66.7%** (`runs/eval_v3_new.txt`). Baselines on the same
split: base model 17.1% exact / 34.1% tool-selection (`runs/eval_v3_base.txt`);
old v2 adapter 24.4% / 56.1% (`runs/eval_v3_old.txt`). See `NOTES.md` /
`runs/eval_v3_summary.md`.

> The previously-listed 65.7% / 77.1% was measured on rows the model had
> TRAINED on (the old trainer's train set included the seed-42 eval holdout),
> so it is NOT comparable to held-out results. The numbers above are the honest,
> disjoint-split baseline.

Runtime invariant: `pane_split` without explicit pane/current targets the
caller's pane; normalization makes it explicit (`current: true`). Eval reports
both raw and normalized accuracy.

## Safety notes

- Eval is side-effect free; it only compares predicted calls to labels.
- Do not close workspaces/tabs/panes/sessions you did not create, and never
  `herdr server stop` from an active session unless intended.
- Command syntax follows the installed `herdr` CLI (v0.8.2), captured under
  `cli_help/` — the installed CLI is the source of truth.
