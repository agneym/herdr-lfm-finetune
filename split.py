"""Deterministic, shared data splits for train and eval.

Single source of truth so the two never disagree:

- The EVAL holdout is carved out first (seed 42, last N% after a full
  shuffle — identical semantics to what eval_lfm2.py always used, so old
  runs/eval_*.txt numbers stay comparable).
- The train-time VALIDATION slice (used for best-checkpoint selection) is
  drawn ONLY from the remaining rows. It can therefore never overlap the
  eval holdout, and checkpoint selection never peeks at eval data.
"""
import random

EVAL_SEED = 42
VAL_SEED = 0


def eval_holdout(n, frac=0.15, seed=EVAL_SEED):
    """Sorted list of the held-out eval row indices."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = max(1, int(n * frac))
    return sorted(idx[-k:])


def train_val(n, val_frac=0.10, eval_frac=0.15,
              seed=VAL_SEED, eval_seed=EVAL_SEED):
    """Return (train_idx, val_idx, eval_idx); the three sets are disjoint."""
    eval_idx = set(eval_holdout(n, eval_frac, eval_seed))
    rest = [i for i in range(n) if i not in eval_idx]
    order = list(rest)
    random.Random(seed).shuffle(order)
    k = max(1, int(n * val_frac))
    val = set(order[:k])
    train = [i for i in rest if i not in val]
    return train, sorted(val), sorted(eval_idx)
