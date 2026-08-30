---
name: lfm2-herdr-train-eval
description: Fine-tune LiquidAI/LFM2-350M as a Herdr terminal-multiplexer expert via PEFT LoRA, train on Google Colab GPUs using the colab CLI, and evaluate locally against a pinned deterministic holdout. Use when training, retraining, or evaluating the Herdr expert model in this repo.
compatibility: Requires this repo's pipeline scripts in the repo root, the `colab` CLI (google-colab-cli) with ADC auth for Colab training, and a local Python venv for eval.
metadata:
  author: agney
  version: "1.1"
---

# LFM2 Herdr Expert — Train & Eval Workflow

Skill for fine-tuning LiquidAI/LFM2-350M as a Herdr terminal-multiplexer
expert, training on Google Colab via `colab` CLI, and evaluating locally.

## Context

- Repo: `herdr-liquid-finetune`. We tried Needle 2 (cactus-needle) first; its
  finetuned adapters produce degenerate output (duplicated keys, garbled args)
  even under ideal conditions — root cause is inside the needle trainer,
  unfixable from outside. We switched to LFM2-350M + PEFT LoRA.
- Data: `dataset.jsonl` is currently **804 rows** (98 off-topic, 12.2%). The
  eval holdout is pinned to **runs/results/eval_v8_holdout.json** (120 rows, keyed by
  query string) so re-eval stays comparable as the dataset grows.

## Current status (v7 adapter, v8 holdout)

Trained with deterministic system-prompt rotation (8 contexts) so the model
grounds workspace/pane/cwd from the prompt instead of memorizing one fixed
`w1:p1 / /home/repo` context. Live-validated against a real `herdr` server
(PASS 404 / SKIP 250 / FAIL 23 — identical profile to v6).

The canonical holdout was re-pinned to **v8** (120 rows) in Phase 1B; the same
**v7** adapter was re-scored on it (strictly disjoint from training):

| model (v8, 120-row holdout) | exact-call | tool-selection | off-topic |
|----------------------------|-----------:|---------------:|----------:|
| base (untuned)             |      6.8% |         25.2% |     47.1% |
| **v7 (current adapter)**   |   **96.1% (99/103)** | **97.1% (100/103)** | **100% (17/17)** |

The prior **98-row numbers are history**: v7 was **96.3%** / v6 **93.9%** on the
v5 holdout; they must NOT be compared head-to-head with v8 (different pin).
Full breakdowns: `runs/results/eval_v8_summary.md`, `runs/results/eval_v7_summary.md`,
`runs/results/eval_v6_summary.md`, etc.

> **Any older published numbers (e.g. 65.7% / 77.1%, 50.0%) are INVALID.** Two
> independent bugs contaminated them: the early trainer built its train set as
> "not in val", which silently INCLUDED the eval holdout; and `eval_lfm2.py`
> used to feed the gold assistant answer back into the prompt. Both are fixed;
> `NOTES.md` has the full story. Never cite the old tables.

## Pipeline files (all in repo root)

| file | purpose |
|---|---|
| `split.py` | SINGLE source of truth for train/val/eval splits; the eval holdout is carved out first and is provably disjoint from training |
| `make_dataset.py` | generates `dataset.jsonl` ({messages, tools, expected}) |
| `train_lfm2.py` | PEFT LoRA SFT; masks loss to assistant tokens only; saves best-val checkpoint |
| `eval_lfm2.py` | holdout eval on the pinned 120 rows; reports raw AND normalized exact-call accuracy |
| `pin_holdout.py` | persists the eval holdout (keyed by query) so re-eval stays comparable as the dataset grows |
| `validate_dataset.py` | live-validates dataset labels against a real `herdr` server |
| `herdr_tools.py` | the Herdr operations; schemas loaded from `reference/herdr_schemas.json` |
| `scripts/run_detached_dump.py` | detached Colab trainer + base64 checkpoint dump |
| `adapters/lfm2_herdr_lora/` | current tuned adapter (weights are on HF Hub as `agney/lfm2-herdr-lora`; `make fetch` pulls them in) |

> `ask_herdr.py` (the NL→operation runtime harness) was written against the
> Needle engine and has been **removed**; see `NOTES.md` "Known gap". Its
> `normalize_call()` invariant lives on in `eval_lfm2.py`.

## Training on Colab (hard-won lessons)

1. Session setup (a T4 is enough for 350M; v7 used an L4):
   ```
   colab new -s NAME --gpu T4
   colab exec -s NAME -f scripts/setup_lfm2_colab.py        # pip transformers>=4.55 peft accelerate (datasets dropped)
   colab exec -s NAME --timeout 400 -f scripts/fix_torchao.py   # pip -U "torchao>=0.16"  (peft 0.20 requires it)
   colab upload -s NAME dataset.jsonl /content/dataset.jsonl
   colab upload -s NAME train_lfm2.py /content/train_lfm2.py
   colab upload -s NAME split.py /content/split.py          # train_lfm2.py imports it
   colab upload -s NAME runs/results/eval_v8_holdout.json /content/eval_v8_holdout.json   # optional: pinned holdout
   ```
2. NEVER run training inside one blocking `colab exec`. Two failure modes:
   - exec timeout kills the run, or
   - the CLI's keep-alive daemon dies when exec returns → Colab idle-prunes
     the VM within minutes and the checkpoint is lost.
   Instead use `scripts/run_detached_dump.py` (nohup-detach the trainer, poll
   train.log every 120 s, keepalive-tick every 60 s, then IMMEDIATELY
   tar+base64-dump the checkpoint into stdout — the VM can be reaped seconds
   after TRAINING OK; this actually happened):
   ```
   colab exec -s NAME -f scripts/run_detached_dump.py --env HOLDOUT=/content/eval_v8_holdout.json
   ```
   (`scripts/watch_and_dump.py` is the no-relaunch companion if training was
   already launched by an exec whose wrapper timed out.)

3. Get the weights. The Colab scripts write the adapter flat at
   `/content/lfm2_herdr_lora`. The canonical source is now the Hub: `make fetch`
   pulls `agney/lfm2-herdr-lora` into `adapters/lfm2_herdr_lora/` (no token for a
   public repo). If you trained a NEW adapter instead, publish it to the Hub and
   update `HF_REPO` in the Makefile. (Legacy: reconstruct the dumped tarball from
   the log with the snippet below, then unpack it.)
   ```python
   import base64, re
   log = open('runs/logs/lfm2v10_dump.log').read()      # the dump log from the colab exec
   m = re.search(r'=== CKPT DUMP START ===\n(.*?)\n=== CKPT DUMP END ===', log, re.S)
   b64 = ''.join(''.join(c for c in l if c.isalnum() or c in '+/=') for l in m.group(1).splitlines())
   open('ckpt.tar.gz','wb').write(base64.b64decode(b64))
   # tar xzf, then strip CLI box-drawing borders if the download failed
   ```

4. Hyperparameters that worked (v7, 804 rows): epochs 12, batch 1, grad-accum 8,
   lr 1e-4, LoRA r=16 alpha=32 on **q/k/v + w1/w3/w2**. Do NOT target
   `out_proj` — it is SHARED with Lfm2ShortConv, and PEFT routes it through
   torchao and crashes. There is **no `o_proj`** at all, so an `o_proj` target
   silently trains q/k/v only.
   - LFM2 naming gotcha: the MLP projections are `w1`/`w3`/`w2`, NOT
     gate/up/down_proj (those match nothing).

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

`pane_split` without explicit pane/current targets the caller's pane.
`normalize_call()` (in `eval_lfm2.py`) injects `current=true`. Eval reports
both raw and normalized accuracy.

## Eval

The published tables are on the **pinned 120-row holdout**
(`runs/results/eval_v8_holdout.json`). Use `--holdout` — it is now REQUIRED (bare runs
are refused):

```
.venv/bin/python eval_lfm2.py --adapter adapters/lfm2_herdr_lora --holdout runs/results/eval_v8_holdout.json
.venv/bin/python eval_lfm2.py --base --holdout runs/results/eval_v8_holdout.json   # baseline
```

Pin the holdout once (keyed by query, so appending training rows never shifts
it) and reuse it on both eval and train. `--out` is required; add `--force` to
overwrite an existing pin — never overwrite the live v5/v8 pins:

```
.venv/bin/python pin_holdout.py --data dataset.jsonl --out runs/results/eval_v9_holdout.json
.venv/bin/python eval_lfm2.py --adapter adapters/lfm2_herdr_lora --holdout runs/results/eval_v8_holdout.json
```

> `--holdout` is REQUIRED (a bare recompute would drift from the pinned set).
> Always pass `--holdout runs/results/eval_v8_holdout.json`. To re-pin again, write a
> NEW versioned file (e.g. eval_v9_holdout.json) and repoint all four live
> consumers + docs in one commit (see `runs/results/caveats.md` 'Re-pinning the holdout').
