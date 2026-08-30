# Eval snapshot — retrained v3 (lfm2v6, clean disjoint split)

Date: 2026-08-24
Checkpoint: adapters/lfm2_herdr_lora (unpacked from ckpt.tar.gz pulled from Google Drive)
  - Drive source: MyDrive/herdr/ckpt-lfm2v6.tar.gz
  - sha256: 39c54ab53350e1a1299d3a097c837f121e8d0215c97043511284c1ff77c6796d
  - size: 2,737,887 bytes
Dataset: dataset.jsonl (314 rows, 27 off-topic) — rebalanced (added rare-tool + agent_wait.until + worktree_create.path examples)
Split: SHARED (split.py, seed 42 eval-holdout / seed 0 val), strictly disjoint: train 236 / val 31 / eval 47
  - Training EXCLUDES the eval holdout rows (fixed the old contamination where train=283 included eval rows).
Hyperparams: epochs 8, batch 1, grad-accum 8, lr 1e-4, LoRA r=16 alpha=32 dropout=0.05 on q/k/v/o_proj, bf16, grad-ckpt.
Train curve: train 3.20 -> 0.23, val 2.24 -> 0.3986 (best at epoch 8; never overfit).

## New adapter on the 47-row holdout (runs/results/eval_v3_new.txt)
  exact-call accuracy  : 13/41 = 31.7%
  exact (normalized)   : 19/41 = 46.3%
  tool-selection acc   : 26/41 = 63.4%
  off-topic (no call)  : 4/6  = 66.7%

## Why this is much lower than the old 65.7%/77.1%
The old adapter (lfm2_herdr_lora_old / runs/results/eval_v2_norm.txt) was trained on rows that
INCLUDED its own eval holdout (old train_lfm2.py made train = all-not-in-val = 283 rows,
which overlapped the seed-42 eval set). It was therefore scored partly on memorized rows,
inflating exact-call and tool-selection. The 47-row set here is genuinely unseen by the new
model, so this is the true generalization number. The new 65.7%-style claim is NOT comparable
to this.

## Dominant failure mode on the holdout
Under-calling: 7 of the top-10 mismatches are `predicted: []` (model emitted NO tool call) on
rows that required a call (e.g. "make a new workspace", "wait until agent reviewer is done",
"create a worktree for branch feature/x"). The model over-generalized the off-topic refusal rule
("answers: []") to novel on-topic phrasings. Minor: 2 wrong-tool / 1 off-topic-called-tool.

## Files
- runs/results/eval_v3_new.txt  — new adapter summary + per-tool grounding + mismatches
- runs/results/eval_v3_base.txt — base baseline on the same split
- runs/results/eval_v3_summary.md — this file

## 3-arm comparison on the SAME 47-row holdout (disjoint, seed 42)
| model | exact-call | exact-norm | tool-select | off-topic |
|---|---|---|---|---|
| base (untuned)                  | 17.1% | 19.5% | 34.1% | 50.0% |
| old adapter v2 (contaminated)   | 24.4% | 39.0% | 56.1% | 83.3% |
| **new adapter v3 (clean split)**| **31.7%** | **46.3%** | **63.4%** | 66.7% |

Conclusions:
- The cleanly-retrained v3 beats the untrained base ~1.85x on exact-call (31.7 vs 17.1) and
  near-2x on tool-selection (63.4 vs 34.1) — retraining WAS worth it.
- v3 also beats the OLD v2 adapter on genuinely-unseen rows (31.7 vs 24.4 exact, 63.4 vs 56.1
  tool-select). This is the key finding: the old 65.7%/77.1% was inflated by train/eval overlap,
  and on the truly-held-out split the old model only manages 24.4% — so the clean retrain is a real
  improvement, not a regression.
- Trade-off: v3 off-topic restraint (66.7%) dropped below old (83.3%) — the new model calls a tool
  on ~1/3 of off-topic rows while old called on ~1/6. That, plus the under-calling on on-topic rows,
  is the next thing to fix (see failure mode below); a few more examples + a tuned off-topic ratio
  should lift both.

Treat 31.7%/63.4% as the true generalization baseline to beat going forward; the old 65.7%/77.1%
is NOT comparable (overlap-contaminated).
