# v8 — re-pinned 120-row holdout

The canonical eval holdout was re-pinned to `runs/results/eval_v8_holdout.json` (Phase 1B
of the repo reorg). `dataset.jsonl` had grown past the v5 pin (804 rows →
`int(804*0.15)=120`), so a fresh pin was carved out (seed 42, keyed by query
string) — the same v7 adapter was re-scored on it to confirm the fine-tune
generalizes past the 98-row set.

This is a **regime note, not a new model**: the adapter is the same `v7`
(`adapters/lfm2_herdr_lora`). These numbers replace the 98-row v5 result as the
current headline; the v6/v7 98-row numbers remain valid history.

## Results — `runs/results/eval_v8_holdout.json` (120 rows, 17 off-topic / 103 on-topic)

| model | exact-call | exact-norm | tool-selection | off-topic |
|---|---:|---:|---:|---:|
| base (untuned) | 6.8% | 6.8% | 25.2% | 47.1% (8/17) |
| **v7 adapter** | **96.1% (99/103)** | **96.1%** | **97.1% (100/103)** | **100% (17/17)** |

The fine-tune holds up on the new holdout: 96.1% exact (vs 96.3% on the 98-row
v5) — within noise — and off-topic refusal is now a perfect 17/17.

Raw outputs: `runs/results/eval_v8_base.txt`, `runs/results/eval_v8_adapter.txt` (full
mismatches + per-tool argument grounding).

## Re-pin (Phase 1B)

- Created `runs/results/eval_v8_holdout.json` via `pin_holdout.py --out`.
- Repointed all four live consumers to v8 in one commit: `Makefile`
  (`eval`/`eval-base`/`train`/`pin-holdout`), `eval_pi.mjs` (`--holdout` default),
  README, SKILL, `runs/results/caveats.md`.
- **`runs/results/eval_v5_holdout.json` (the 98-row v6/v7 pin) was left untouched** —
  its references in `eval_v5/v6/v7_summary.md` and `eval_v6/v7_new.log` are
  history and must not be rewritten.

## Frontier-model comparison

The pi-harness frontier comparison (deepseek / GLM flash) was scored on the
98-row v5 holdout and is NOT re-run on v8 here (paid API). Those numbers stand
as history (see `runs/results/eval_deepseek_summary.md`, `runs/results/eval_glm_summary.md` /
README); the v8 frontier run is pending.
