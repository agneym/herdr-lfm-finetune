# Notes

## Why this is not a Needle 2 repo anymore

The original approach fine-tuned **Needle 2** (`cactus-needle`, 45M) on Herdr
tool schemas with its grammar-constrained decoding. We abandoned it in
commit `c2a2070` ("Switch Herdr expert from Needle 2 to LFM2-350M LoRA").

What we saw: the finetuned Needle adapters produce **degenerate output**
(duplicated JSON keys, garbled arguments) even under ideal training conditions.
The root cause is inside the needle trainer itself and was not fixable from
outside. On the same holdout split (seed 42):

| model                        | exact-call | tool-selection |
|------------------------------|-----------:|---------------:|
| Needle base                  |      18.5% |            n/a |
| Needle LoRA tuned (v2)       |      11.1% |          37.0% |
| LFM2 base                    |       7.4% |          22.2% |
| **LFM2 + LoRA (current)**    | **65.7% raw / 77.1% normalized** | **97.1%** |

Tuning made Needle *worse* than its own baseline; the same data makes LFM2-350M
near-saturated on tool selection.

## Going back to the Needle track

The full Needle pipeline as it stood just before the switch is commit
**`308aa1d`** ("Created using Colab"). To revisit it:

```sh
git checkout 308aa1d -- make_dataset.py train_herdr_agent.py eval_model.py \
    ask_herdr.py colab_herdr_finetune.ipynb
```

(`data.jsonl` and `herdr_tools.py` were never deleted — they are still the
front half of the LFM pipeline.)

## Shared files that mention needle

Two kept files still import the `needle` package:

- `herdr_tools.py` — uses `needle.agent.tools.build_schema` only to derive the
  25 Herdr op schemas that end up in `data.jsonl`. The functions themselves are
  plain `herdr` CLI wrappers.
- `make_dataset.py` — imports `herdr_tools.SCHEMAS` to generate `data.jsonl`.

So regenerating `data.jsonl` requires `cactus-needle` installed, but training
and inference (`train_lfm2.py`, `eval_lfm2.py`) are pure transformers+PEFT.
If you want to drop the needle dependency entirely, extract the schemas to a
static JSON file once (`python -c "import json, herdr_tools as ht;
json.dump(ht.SCHEMAS, open('herdr_schemas.json','w'))"`) and point
`make_dataset.py` at it.

## Known gap

`ask_herdr.py` (the natural-language -> operation runtime harness Hermes calls)
was written against the Needle engine and was removed with the old track; the
version at commit `c2a2070` is the last one. Porting it means loading the LFM2
LoRA adapter and reusing the `normalize_call()` invariant (also implemented in
`eval_lfm2.py`). Until then, `eval_lfm2.py` shows the full planner loop.
