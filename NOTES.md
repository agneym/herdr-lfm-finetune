# Notes

## Why this is not a Needle 2 repo anymore

The original approach fine-tuned **Needle 2** (`cactus-needle`, 45M) on Herdr
tool schemas with its grammar-constrained decoding. We abandoned it in
commit `c2a2070` ("Switch Herdr expert from Needle 2 to LFM2-350M LoRA").

What we saw: the finetuned Needle adapters produce **degenerate output**
(duplicated JSON keys, garbled arguments) even under ideal training conditions.
The root cause is inside the needle trainer itself and was not fixable from
outside.

## IMPORTANT: the old eval numbers were contaminated

The previous table (LFM2 + LoRA **65.7% / 77.1%**, tool-selection 97.1%) was
measured on rows the model had TRAINED on. The old `train_lfm2.py` built its
train set as "everything not in val" (283 rows), which silently INCLUDED the
seed-42 eval holdout. So the model was scored partly on memorized queries, and
the number is not comparable to any held-out result.

**The honest, held-out numbers (strictly disjoint split, seed 42, 47 rows):**

| model                        | exact-call | tool-selection | off-topic |
|------------------------------|-----------:|---------------:|----------:|
| LFM2 base                    |      17.1% |          34.1% |     50.0% |
| LFM2 + LoRA (old v2)         |      24.4% |          56.1% |     83.3% |
| **LFM2 + LoRA (v3, clean)**  |   **31.7%** |      **63.4%** |     66.7% |

- The clean v3 retrain beats the untrained base ~1.85x on exact-call and ~2x
  on tool-selection, AND beats the old v2 adapter on genuinely-unseen rows
  (31.7 vs 24.4) — so the retrain is a real improvement, not a regression.
- The v3 model under-calls on some on-topic rows (predicts no call) and calls a
  tool on ~1/3 of off-topic rows; that's the next thing to fix (more examples,
  tuned off-topic ratio / refusal phrasing).
- Full numbers and per-tool grounding:
  `runs/eval_v3_new.txt` (v3), `runs/eval_v3_base.txt` (base),
  `runs/eval_v3_old.txt` (old v2), `runs/eval_v3_summary.md` (3-arm table).

## Split discipline going forward

Always carve the eval holdout OUT of training (a disjoint split). `split.py`
owns this now: `eval_holdout()` (seed 42, shuffle-last-N%) is called first, then
`train_val()` draws the val slice only from the remainder. Both `train_lfm2.py`
and `eval_lfm2.py` import it, so the eval rows can never leak into training.

## Going back to the Needle track

The full Needle pipeline as it stood just before the switch is commit
**`308aa1d`** (\"Created using Colab\"). To revisit it:

```sh
git checkout 308aa1d -- make_dataset.py train_herdr_agent.py eval_model.py \
    ask_herdr.py colab_herdr_finetune.ipynb
```

(`data.jsonl` and `herdr_tools.py` were never deleted — they are still the
front half of the LFM pipeline.)

## Shared files that mention needle

The canonical tool schemas are now a static file,
`reference/herdr_schemas.json` (dumped once from `herdr_tools.py` when the
repo still used the needle toolkit). `herdr_tools.py` loads them from there —
the repo no longer needs `cactus-needle` installed. If you change a tool's
signature, update both the function and the JSON entry (or temporarily
reinstall cactus-needle and re-dump).

## Known gap

`ask_herdr.py` (the natural-language -> operation runtime harness Hermes calls)
was written against the Needle engine and was removed with the old track; the
version at commit `c2a2070` is the last one. Porting it means loading the LFM2
LoRA adapter and reusing the `normalize_call()` invariant (also implemented in
`eval_lfm2.py`). Until then, `eval_lfm2.py` shows the full planner loop.
