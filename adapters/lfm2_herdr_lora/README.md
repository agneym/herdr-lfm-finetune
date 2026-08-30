---
base_model: LiquidAI/LFM2-350M
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:LiquidAI/LFM2-350M
- lora
- transformers
- herdr
license: mit
---

# LFM2-Herdr Expert (LoRA adapter)

A PEFT LoRA adapter over `LiquidAI/LFM2-350M`, fine-tuned to be an expert on the
[Herdr](https://herdr.dev) terminal multiplexer: given a natural-language
request, it emits the correct Herdr tool call (or refuses off-topic prompts).
This is a **narrow specialist** — it plans the 25 Herdr operations, not a
general chat/code/reasoning model.

## Loading

Load it on top of the base model with `peft`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")
model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2-350M", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "agneym/lfm2-herdr-lora").eval()

prompt = tok.apply_chat_template(
    [{"role": "system", "content": "HERDR_ENV=1\nworkspace=w1\ntab=w1:t1\npane=w1:p1\ncwd=/home/repo\nagent kind=hermes"},
     {"role": "user", "content": "split my pane"}],
    tools=..., tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=192, do_sample=False)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=False))
```

The model answers in native `<|tool_call_start|>[name(k=v, ...)]<|tool_call_end|>`
syntax. Load the tool schemas from
[`reference/herdr_schemas.json`](https://github.com/agneym/herdr-lfm-finetune)
in the training repo.

## Evaluation

Scored on the pinned 120-row holdout (`runs/results/eval_v8_holdout.json`,
seed 42, strictly disjoint from training), all 25 tools represented:

| model | exact-call | tool-selection | off-topic |
|---|---:|---:|---:|
| base (untuned) | 6.8% | 25.2% | 47.1% (8/17) |
| **this adapter** | **96.1% (99/103)** | **97.1% (100/103)** | **100% (17/17)** |

`exact-call` requires the tool name and arguments to match the label
(key-order-insensitive, `pane_split` normalized to `current=true`).

## Training

- Base: `LiquidAI/LFM2-350M` (bf16, gradient checkpointing), T4/L4.
- Data: 804 rows in `dataset.jsonl` (98 off-topic, 12.2%), system-prompt
  rotation over 8 contexts so grounding comes from the prompt, not a memorized
  `w1:p1 / /home/repo` constant.
- LoRA: `r=16`, `alpha=32`, dropout 0.05, targets `q_proj/k_proj/v_proj/w1/w3/w2`
  (the LFM2 MLP projections are `w1/w3/w2`, not `gate/up/down_proj`; do NOT
  target `out_proj`, which is shared with `Lfm2ShortConv`).
- SFT: 12 epochs, batch 1, grad-accum 8, lr 1e-4, cosine schedule, loss masked
  to assistant tokens only, best-val checkpoint.

## Limitations

- Fails some novel paraphrases ("give me a new pane on the right" →
  `pane_create(Direction=...)`; "where am i?" under-calls).
- `pane_split` and `pane_current` argument grounding is below 100% on the
  holdout.
- Does not do general chat/code/reasoning; it plans the 25 Herdr ops and
  refuses off-topic prompts.

## License

MIT. The full pipeline (dataset generation, training, eval) is in
[`herdr-liquid-finetune`](https://github.com/agneym/herdr-lfm-finetune).
