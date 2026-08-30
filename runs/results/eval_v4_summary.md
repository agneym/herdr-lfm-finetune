# Eval snapshot — v4 (lfm2v7, semantic-refusal dataset + MLP LoRA targets)

Date: 2026-08-24 (trained ~6h on Colab T4, session lfm2v7)
Checkpoint: adapters/lfm2_herdr_lora
  - source: runs/checkpoints/ckpt-lfm2v7.tar.gz (dumped from Colab)
  - sha256: 51cdd14e915d83d59f39fc34e8bef8e464020fd24cc7cd88a014299de3986696
Dataset: dataset.jsonl v4 — 656 rows, 79 off-topic (12.0%)
  - >=10 surface forms per tool (was 4-7 for tail tools)
  - hard-negative off-topic (Herdr verbs, non-Herdr actions) -> semantic refusal boundary
  - contrastive minimal pairs (worktree list/create, agent get/install, pane read/wait)
Split: SHARED (split.py seed 42), train 493 / val 65 / eval 98 (16 off-topic);
  all 25 tools have >=1 eval row. NOT comparable to the v3 47-row table.
Trainer changes: epochs 12, LoRA r=16 alpha=32 on q/k/v_proj + w1/w3/w2.
  Discovery: LFM2 has no o_proj; old "o_proj" target was a silent no-op, so v3
  trained q/k/v only (~0.9M params). v4 trains ~4.8M incl. all MLP projections.
Val curve: 0.693, 0.226, 0.111, 0.127, 0.063, 0.050, 0.032(best@7... 0.031 best@9),
  best checkpoint at epoch 9 val 0.0310.

## v4 adapter on the 98-row holdout (runs/results/eval_v4_new.log)
  exact-call accuracy  : 41/82 = 50.0%
  exact (normalized)   : 41/82 = 50.0%
  tool-selection acc   : 53/82 = 64.6%
  off-topic (no call)  : 16/16 = 100.0%

## vs v3 (different split — directionally comparable only)
| metric        | v3 (47-row) | v4 (98-row) |
|---------------|-------------|-------------|
| exact-call    | 31.7%       | 50.0%       |
| exact-norm    | 46.3%       | 50.0%       |
| tool-select   | 63.4%       | 64.6%       |
| off-topic     | 66.7%       | **100.0%**  |

## 2-arm comparison on the SAME 98-row holdout
| model | exact-call | exact-norm | tool-select | off-topic |
|---|---|---|---|---|
| base (untuned)              | 20.7% | 20.7% | 34.1% | 56.2% |
| **v4 adapter (clean split)**| **50.0%** | **50.0%** | **64.6%** | **100.0%** |

v4 beats base 2.4x on exact-call and 1.9x on tool-selection, with perfect
off-topic restraint. Base also invents tools (`list_worktrees`), which the
tuned adapter never does.

The headline fixes landed:
- off-topic restraint 66.7% -> 100%: hard negatives taught a semantic boundary;
  zero refusals leaked AND (unlike naive ratio-tuning) on-topic recall improved.
- exact-call +18pts: paraphrase expansion fixed under-calling on novel phrasings
  (worktree_list went 0/3 -> 7/7).

## Remaining failure modes (next lever)
1. pane_split 0/5 exact — ALL five misses are `predicted: []` (under-call), even
   though pane_split has the most training rows (43). Suspicion: label collision
   with the off-topic-style reasoning line or over-triangulated arg combos
   (cwd+focus+ratio variants) making the marginal distribution too wide; needs a
   look at what the model emits instead (all-empty = refusal bias, wrong-tool =
   confusion).
2. Optional-arg omission: "read the last 120 lines of w1:p1" -> dropped `lines`.
   Suggests more explicit-arg contrast pairs (with/without lines).
3. Residual verb confusions: tab_create->tab_list, pane_run->pane_read.

## Files
- runs/results/eval_v4_new.log  — v4 adapter summary + grounding + mismatches
- runs/results/eval_v4_base.log — base baseline on same split
- runs/logs/lfm2v7_dump.log  — training log + checkpoint dump
- runs/checkpoints/ckpt-lfm2v7.tar.gz — raw checkpoint tarball

Treat 50.0%/64.6%/100% as the baseline to beat; v3 numbers are kept for
directional context but are not apples-to-apples (different split).
