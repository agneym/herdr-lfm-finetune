# Eval snapshot — v6 (targeted fixes for the 5 v5 failure classes)

Date: 2026-08-27 (Colab L4, session lfm2v9)
Checkpoint: adapters/lfm2_herdr_lora
  - source: runs/ckpt-lfm2v9.tar.gz (reconstructed from runs/lfm2v9_dump.log)
  - sha256: 94c35c3b9c0833a3252180ffe155ef5c3287da4e9decd1556a0a166a128937d2
Dataset: dataset.jsonl v6 — 804 rows, 98 off-topic (12.2%)
  - Same 98-row pinned holdout as v5 (runs/eval_v5_holdout.json, keyed by
    query string), so directly comparable to v5.  All v6 rows are append-only;
    original 656 rows untouched (holdout indices all < 656).
  - +129 on-topic rows targeting the v5 failure classes; +19 off-topic to hold
    the ~12% ratio.
Trainer: hyperparams unchanged from v5 (epochs 12, batch 1, grad-accum 8,
  lr 1e-4, LoRA r=16 alpha=32 on q/k/v_proj + w1/w3/w2).  Ran on L4
  (~1 h vs ~6 h on the free-tier T4 — note the "~15 min" in older notes was
  wrong; T4 is ~21 min/epoch at this size).
Val curve (partial — poll snapshots missed some epochs): 0.4537, 0.1778, …,
  0.0815, …, 0.0505, 0.0525, …, 0.0257, 0.0249, …, 0.0249 — best val 0.0249
  (v5 was 0.0251).

## v6 vs v5 on the SAME 98-row holdout (pinned, seed 42)

| model | exact-call | exact-norm | tool-selection | off-topic |
|---|---:|---:|---:|---:|
| base (untuned)              | 9.8%  | 9.8%  | 26.8% | 68.8% |
| v4 (bare format)            | 70.7% | 75.6% | 91.5% | 100.0% |
| v5 (native format)          | 85.4% | 85.4% | 95.1% | 100.0% |
| **v6 (targeted fixes)**     | **93.9%** | **93.9%** | **96.3%** | **100.0%** |

v6 = +8.5 pts exact-call, +1.2 pts tool-selection over v5, with zero
normalization (native format, same as v5).

## v5 failures fixed (9 of 12)

| v5 row | failure | v6 |
|---|---|---|
| 32  | pane_read dropped `source=visible` | fixed (pane_read 3/3) |
| 44  | agent_get said `pane=` not `target=` | fixed (agent_get 2/2) |
| 103 | "horizontally" -> `down` | fixed (only pane_split miss is now row 104) |
| 166/167 | "show git status" -> pane_list | fixed (pane_run 7/7) |
| 217 | `kind=triage` instead of `hermes` | fixed (agent_start 7/7) |
| 220 | "120s" -> `120` not `120000` ms | fixed (agent_start 7/7) |
| 223/224 | agent_read dropped `lines` | fixed (agent_read 3/3) |

## Remaining failure modes (5 — next levers for v7)

1. **pane_wait regex + timeout arg separation (row 196, persists).** The key is
   now correct (`regex`), but the model still emits
   `regex='…(succeeded|failed)' timeout_ms=90000` (a space where the comma
   should be) — the `(…|…)` in the pattern derails the arg list.  parse_calls
   then folds `timeout_ms=90000` into the regex value.
2. **pane_current "where am i?" (rows 388/389, persists).** Now *under*-calls
   (emits no tool call) instead of emitting a spurious arg.  Both "where am i?"
   surface forms are held out; training has none.
3. **agent_wait "hold until agent triage reports done" (row 574, new).** Parses
   `triage reports` as the target instead of stripping "reports done" ->
   target=triage, until=[done].  The triage variant is the held-out one.
4. **pane_split "give me a new pane on the right" (row 104, new/regression).**
   Emits `pane_current` instead of `pane_split(current=true, direction=right)`.
   "give me a new pane" is a create/split verb, but only the "below" variant is
   in training.

## Files
- runs/eval_v6_new.log      — v6 adapter on the pinned holdout
- runs/lfm2v9_dump.log      — training log + base64 checkpoint dump
- runs/ckpt-lfm2v9.tar.gz   — raw checkpoint tarball (gitignored)
- runs/eval_v5_holdout.json — the pinned 98-row holdout (keyed by query)
