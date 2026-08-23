# Herdr expert: LFM2-350M LoRA

Fine-tune **LiquidAI/LFM2-350M** (PEFT LoRA) to be an expert on the **Herdr**
terminal multiplexer, so a main agent can ask it in natural language for the
right Herdr operation and run the result.

> An earlier attempt used Needle 2 (cactus-needle). It produced degenerate
> output (duplicated keys, garbled args) from its trainer and was abandoned —
> see `NOTES.md` for the numbers and the commit to go back to.

---

## How it works

The pipeline is four steps:

1. `make_dataset.py` generates `data.jsonl` — 270 examples of
   `{system, tools, query, reasoning, answers}` covering all 25 Herdr ops,
   ~12% off-topic. The tool schemas come from `herdr_tools.py`.
2. `make_lfm2_dataset.py` converts it to chat format: each row of
   `data_lfm2.jsonl` is `{messages, tools}` ready for
   `tokenizer.apply_chat_template(tools=...)`. Assistant turns keep the
   reasoning line, then emit calls in native LFM2 syntax `[name(arg=val)]`;
   off-topic rows learn a text-only refusal.
3. `train_lfm2.py` — PEFT LoRA SFT (loss masked to assistant tokens only),
   saves the best-validation checkpoint.
4. `eval_lfm2.py` — scores on a deterministic holdout (seed 42, last 15%),
   reporting raw AND normalized exact-call accuracy.

## Files

| File | Purpose |
|------|---------|
| `herdr_tools.py` | The 25 Herdr operations + ground-truth schemas (needs `cactus-needle` only for schema building). |
| `make_dataset.py` | Generates `data.jsonl`. |
| `make_lfm2_dataset.py` | `data.jsonl` -> chat-format `data_lfm2.jsonl`. |
| `train_lfm2.py` | LoRA SFT driver (run on Colab GPU; see below). |
| `eval_lfm2.py` | Holdout eval; `--base` for baseline. |
| `validate_dataset.py` | Live-validates dataset labels against a real `herdr` server. |
| `lfm2_herdr_lora/` | Current tuned adapter. `lfm2_herdr_lora_v1/` is the previous run. |
| `NOTES.md` | Why we dropped Needle 2; how to recover that track. |
| `cli_help/` | Captured `herdr` CLI help texts (dataset source of truth). |

Everything else (`run_detached_*.py`, `setup_lfm2_colab.py`, `fix_torchao.py`,
`*_dump*.log`, `eval_lfm2_*.txt`, ...) is Colab-glue and experiment artifacts;
see `.gitignore`.

## Train on Google Colab

A T4 is enough for 350M (~15 min). Use the `colab` CLI:

```sh
colab new -s NAME --gpu T4
colab exec -s NAME -f setup_lfm2_colab.py            # transformers>=4.55 peft datasets accelerate
colab exec -s NAME --timeout 400 -f fix_torchao.py   # torchao>=0.16 (peft 0.20 requires it)
colab upload -s NAME data_lfm2.jsonl /content/data_lfm2.jsonl
colab upload -s NAME train_lfm2.py /content/train_lfm2.py
```

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
.venv/bin/python eval_lfm2.py --adapter lfm2_herdr_lora --split 0.15
.venv/bin/python eval_lfm2.py --base          # baseline
```

Current numbers (40-row holdout, seed 42): exact-call **65.7% raw / 77.1%
normalized**, tool-selection **97.1%**, off-topic restraint **100%**
(`eval_lfm2_v2_norm.txt`). Baselines: base model 7.4% exact / 22.2%
tool-selection (`eval_lfm2_base.txt`); first adapter version 40.7% /
92.6% (`eval_lfm2_full.txt`).

Runtime invariant: `pane_split` without explicit pane/current targets the
caller's pane; normalization makes it explicit (`current: true`). Eval reports
both raw and normalized accuracy.

## Safety notes

- Eval is side-effect free; it only compares predicted calls to labels.
- Do not close workspaces/tabs/panes/sessions you did not create, and never
  `herdr server stop` from an active session unless intended.
- Command syntax follows the installed `herdr` CLI (v0.8.2), captured under
  `cli_help/` — the installed CLI is the source of truth.
