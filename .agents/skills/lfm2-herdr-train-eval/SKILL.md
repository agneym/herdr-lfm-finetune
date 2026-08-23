---
name: lfm2-herdr-train-eval
description: Fine-tune LiquidAI/LFM2-350M as a Herdr terminal-multiplexer expert via PEFT LoRA, train on Google Colab GPUs using the colab CLI, and evaluate locally against a deterministic holdout split. Use when training, retraining, or evaluating the herdr needle model in this repo.
compatibility: Requires this repo's pipeline scripts in the repo root, the `colab` CLI (google-colab-cli) with ADC auth for Colab training, and a local Python venv for eval.
metadata:
  author: agney
  version: "1.0"
---

# LFM2 Herdr Expert — Train & Eval Workflow

Skill for fine-tuning LiquidAI/LFM2-350M as a Herdr terminal-multiplexer
expert, training on Google Colab via `colab` CLI, and evaluating locally.

## Context

- Repo: herdr-needle-research. Original stack was cactus-needle (Needle 2);
  its finetuned adapters produce degenerate output (duplicated keys, garbled
  args) even under ideal conditions — root cause is inside the needle
  trainer, unfixable from outside. We switched to LFM2-350M + PEFT LoRA.
- Results (33→40 row holdout, same seed): Needle base 18.5% exact /
  LFM2 base 7.4% → LFM2 LoRA **65.7% raw / 77.1% with runtime normalization**,
  97.1% tool-selection, 100% off-topic restraint.

## Pipeline files (all in repo root)

| file | purpose |
|---|---|
| `make_dataset.py` | generates `data.jsonl` (query/reasoning/answers/tools/system) |
| `make_lfm2_dataset.py` | converts to chat format: `data_lfm2.jsonl` rows = {messages, tools} |
| `train_lfm2.py` | PEFT LoRA SFT; masks loss to assistant tokens only; saves best-val checkpoint |
| `eval_lfm2.py` | same holdout split as old eval_model.py (seed 42, last 15%); reports raw AND normalized exact-call accuracy |
| `ask_herdr.py` | runtime harness; includes `normalize_call()` invariant |

## Training on Colab (hard-won lessons)

1. Session setup (T4 is enough for 350M):
   ```
   colab new -s NAME --gpu T4
   colab exec -s NAME -f setup_lfm2_colab.py        # pip transformers>=4.55 peft datasets accelerate
   colab exec -s NAME --timeout 400 -f fix_torchao.py   # pip -U "torchao>=0.16"  (peft 0.20 requires it)
   colab upload -s NAME data_lfm2.jsonl /content/data_lfm2.jsonl
   colab upload -s NAME train_lfm2.py /content/train_lfm2.py
   ```

2. NEVER run training inside one blocking `colab exec`. Two failure modes:
   - exec timeout kills the run, or
   - the CLI's keep-alive daemon dies when exec returns → Colab idle-prunes
     the VM within minutes and the checkpoint is lost.
   Instead use `run_detached_dump.py`: nohup-detach the trainer, poll
   train.log every 120 s, keepalive-tick every 60 s, and on completion
   IMMEDIATELY tar+base64-dump the checkpoint into stdout (the VM can be
   reaped seconds after TRAINING OK — this actually happened).

3. Reconstruct locally:
   ```python
   import base64, re, tarfile
   log = open('lfm2_train_dumpN.log').read()
   m = re.search(r'=== CKPT DUMP START ===\n(.*?)\n=== CKPT DUMP END ===', log, re.S)
   b64 = ''.join(''.join(c for c in l if c.isalnum() or c in '+/=') for l in m.group(1).splitlines())
   open('ckpt.tar.gz','wb').write(base64.b64decode(b64))
   # tar xzf, then strip CLI box-drawing borders if download failed
   ```

4. Hyperparameters that worked (270 rows): epochs 8, batch 1, grad-accum 8,
   lr 1e-4, LoRA r=16 alpha=32 on q/k/v/o_proj ONLY. Do NOT target conv
   in_proj/out_proj — PEFT routes them through torchao and crashes.

5. Batch 2 OOMs a T4 at ~2.7k-token sequences; batch 1 + gradient
   checkpointing fits.

## Transformers 5.x gotchas

- `apply_chat_template(tokenize=True)` returns a `BatchEncoding`, which is
  NOT an isinstance-dict but iterates like one: `list(be)` yields
  ['input_ids','attention_mask'] (strings!). Unwrap with
  `hasattr(ids,'keys') and 'input_ids' in ids`.
- `torch_dtype=` deprecated → `dtype=`.
- Generation: temperature=None, top_p=None, top_k=None explicitly to disable
  sampling warnings under do_sample=False.

## Runtime invariant

pane_split without explicit pane/current targets caller's pane.
`normalize_call()` in ask_herdr.py injects current=true. Eval reports both
raw and normalized accuracy.

## Eval

```
.venv/bin/python eval_lfm2.py --adapter lfm2_herdr_lora --split 0.15
```
Baseline: `--base`. The split is deterministic (seed 42, last 15%) so runs
are comparable across dataset versions (row count changes with the dataset).
