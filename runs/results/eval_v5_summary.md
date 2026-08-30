# Eval snapshot — v5 (native tool-call format, honest disjoint-split eval)

Date: 2026-08-27 (Colab T4, session lfm2v8)
Checkpoint: adapters/lfm2_herdr_lora
  - source: runs/checkpoints/ckpt-lfm2v8.tar.gz (reconstructed from runs/logs/lfm2v8_dump.log)
  - mirror: Google Drive MyDrive/herdr/ckpt-lfm2v8.tar.gz
  - sha256: d74731362fd02e4fe0ba2d81341bcbd7e914b7a38a98ec1d90f3591cd0230e33
Dataset: dataset.jsonl v5 — 656 rows, 79 off-topic (12.0%)
  - SAME rows/coverage as v4, but the assistant turn is now emitted as a
    structured `tool_calls` field so the LFM2 chat template renders the NATIVE
    syntax `reasoning<|tool_call_start|>[name(k=v, ...)]<|tool_call_end|>`.
  - v4 hand-rendered bare `[name(...)]` and DROPPED the key of every non-string
    arg (`current` -> bare `true`), so calls were malformed and the model
    learned positional args (and, for several tools, stopped after the
    reasoning line entirely — EOS at the `\n[` boundary).
Trainer: epochs 12, batch 1, grad-accum 8, lr 1e-4, LoRA r=16 alpha=32 on
  q/k/v_proj + w1/w3/w2 (unchanged from v4).
Val curve: 0.3984, 0.1532, 0.0833, 0.0625, 0.0591, 0.0301, 0.0285, 0.0264,
  0.0255, 0.0254, 0.0251, 0.0253 — best epoch 11 (val 0.0251, vs v4's 0.0310).

## IMPORTANT — the old eval numbers were invalid (now fixed)

eval_lfm2.py's ask() rendered `row["messages"]` (system + user + the GOLD
assistant answer) with add_generation_prompt=True. Every prior number (v1-v4)
therefore fed the model its own answer and scored its *continuation*, not its
ability to map query -> call. Fixed: prompt is now `row["messages"][:-1]`
(matching train_lfm2.py's masking). All numbers below are honest.

## 3-way comparison on the SAME 98-row holdout (seed 42, strictly disjoint)

| model | exact-call | exact-norm | tool-selection | off-topic |
|---|---:|---:|---:|---:|
| base (untuned)              | 9.8%  | 9.8%  | 26.8% | 68.8% |
| v4 (bare format)            | 70.7% | 75.6% | 91.5% | 100.0% |
| **v5 (native format)**      | **85.4%** | **85.4%** | **95.1%** | **100.0%** |

v5 = +14.7pts exact, +3.6pts tool-selection over v4. v4 needs normalization
(75.6 vs 70.7 raw) to recover dropped arg keys; v5 is exact with zero
normalization — the key=value rendering fix is the whole difference.

## v5 per-tool grounding (exact)
  agent_get 1/2, agent_list 2/2, agent_read 1/3, agent_start 5/7,
  agent_wait 5/5, herdr_status 4/4, integration_install 3/3, pane_close 1/1,
  pane_current 1/3, pane_layout 6/6, pane_list 3/3, pane_read 2/3,
  pane_rename 2/2, pane_run 5/7, pane_send_keys 1/1, pane_split 4/5,
  pane_wait 0/1, session_list 2/2, tab_create 4/4, tab_list 3/3,
  workspace_create 1/1, workspace_get 2/2, workspace_list 2/2,
  worktree_create 3/3, worktree_list 7/7.

## Remaining failure modes (next levers)
1. Optional-arg omission: `source=visible` (row 32), `lines=N` (rows 223/224).
2. Key-name confusion: agent_get `target` vs `pane` (row 44).
3. Direction ambiguity: "horizontally" -> down vs right (row 103).
4. Verb confusion: "show git status" -> pane_list instead of pane_run (166/167).
5. Unit conversion: "120s" -> `120` not `120000` ms (row 220); "kind" =
   `triage` instead of `hermes` (row 217).
6. pane_wait key + arg-separation error (row 196) — the MODEL emits
   `match='Build (succeeded|failed)'(timeout_ms=90000)`: wrong key (`match`
   vs `regex`) and `(timeout_ms=...)` glued to the string with no comma.
   parse_calls handles the well-formed form correctly, so this is a model
   error, not a parser bug.

## Files
- runs/results/eval_v5_holdout.json — pinned 98-row eval holdout (keyed by query)
- runs/results/eval_v5_new.log     — v5 adapter (honest eval)
- runs/results/eval_v4_honest.log  — v4 adapter on the fixed eval
- runs/results/eval_base_honest.log — untuned base on the fixed eval
- runs/logs/lfm2v8_dump.log     — training log + base64 checkpoint dump
- runs/checkpoints/ckpt-lfm2v8.tar.gz  — raw checkpoint tarball (gitignored)
