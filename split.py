"""Deterministic, shared data splits for train and eval.

Single source of truth so the two never disagree:

- The EVAL holdout is carved out first (seed 42, last N% after a full
  shuffle — identical semantics to what eval_lfm2.py always used, so old
  runs/results/eval_*.txt numbers stay comparable).
- The train-time VALIDATION slice (used for best-checkpoint selection) is
  drawn ONLY from the remaining rows. It can therefore never overlap the
  eval holdout, and checkpoint selection never peeks at eval data.
"""
import json
import random

EVAL_SEED = 42
VAL_SEED = 0


def eval_holdout(n, frac=0.15, seed=EVAL_SEED):
    """Sorted list of the held-out eval row indices."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = max(1, int(n * frac))
    return sorted(idx[-k:])


def train_val_with_eval(n, eval_idx, val_frac=0.10, seed=VAL_SEED):
    """Return (train_idx, val_idx, eval_idx) for a GIVEN eval_idx set, with
    the val slice drawn only from rows outside it. The three sets are
    disjoint. Used when the eval holdout is pinned to a file."""
    eval_idx = set(eval_idx)
    rest = [i for i in range(n) if i not in eval_idx]
    order = list(rest)
    random.Random(seed).shuffle(order)
    k = max(1, int(n * val_frac))
    val = set(order[:k])
    train = [i for i in rest if i not in val]
    return train, sorted(val), sorted(eval_idx)


def train_val(n, val_frac=0.10, eval_frac=0.15,
              seed=VAL_SEED, eval_seed=EVAL_SEED):
    """Return (train_idx, val_idx, eval_idx); the three sets are disjoint."""
    eval_idx = eval_holdout(n, eval_frac, eval_seed)
    return train_val_with_eval(n, eval_idx, val_frac, seed)


def save_pinned_holdout(path, rows, frac=0.15, seed=EVAL_SEED):
    """Persist the current eval holdout to `path`, keyed by query string.

    Keying by query (unique per row) instead of index means the pinned eval
    rows stay put when the dataset grows — appending training rows does not
    shift them. Returns the holdout indices.
    """
    idx = eval_holdout(len(rows), frac, seed)
    queries = [rows[i]["messages"][1]["content"] for i in idx]
    meta = {"seed": seed, "frac": frac, "n": len(rows), "queries": queries}
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return idx


def load_pinned_holdout(path, rows):
    """Load a pinned holdout; return (indices, meta).

    Matches rows by query string. Raises if a pinned query is missing or the
    dataset contains duplicate queries (row identity would be ambiguous).
    """
    meta = json.load(open(path))
    by_query = {}
    for i, r in enumerate(rows):
        q = r["messages"][1]["content"]
        if q in by_query:
            raise ValueError(f"duplicate query in dataset: {q!r}")
        by_query[q] = i
    idx = []
    for q in meta["queries"]:
        if q not in by_query:
            raise KeyError(f"pinned holdout query not in dataset: {q!r}")
        idx.append(by_query[q])
    return sorted(idx), meta
