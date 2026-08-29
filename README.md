# Herdr expert: LFM2-350M LoRA

Fine-tune **LiquidAI/LFM2-350M** (PEFT LoRA) to be an expert on the **Herdr**
terminal multiplexer, so a main agent can ask it in natural language for the
right Herdr operation and run the result.

![Herdr holdout accuracy: base vs fine-tune vs frontier models](docs/eval_comparison.png)

*A 350M LoRA fine-tune turns a 9.8% base model into a 96.3% expert — beating
deepseek flash (56.1%) and GLM 5.3 flash (73.2%) on Herdr tool-calling, on the
pinned 98-row holdout.*

**Current state:** the tuned adapter is `adapters/lfm2_herdr_lora` (the "v7"
run). Current dataset is 804 rows (98 marked off-topic — 12.2%) across all 25
Herdr ops. See `runs/eval_v7_summary.md` for the run history and per-version
comparisons.

---

## How it works

The pipeline is three steps:

1. `make_dataset.py` generates `dataset.jsonl` — `{messages, tools, expected}`
   entries covering all 25 Herdr ops, ~12% off-topic including hard negatives
   that reuse Herdr verbs but act outside Herdr. Tail tools get >=10 surface
   forms; contrastive minimal pairs separate confusable ops (list/create
   worktrees, get/install). The tool schemas come from `herdr_tools.py`.
   `messages` are chat-format, ready for
   `tokenizer.apply_chat_template(tools=...)`: on-topic assistant turns keep
   the reasoning line and carry a structured `tool_calls` field, which the
   chat template renders in native
   `<|tool_call_start|>[name(k=v, ...)]<|tool_call_end|>` syntax; off-topic
   rows learn a natural-language refusal. `expected` carries the structured
   tool-call labels for validate/eval.
2. `train_lfm2.py` — PEFT LoRA SFT (loss masked to assistant tokens only),
   saves the best-validation checkpoint (training runs on a Colab GPU — see below).
3. `eval_lfm2.py` — scores against a pinned, query-keyed holdout, reporting raw
   AND normalized exact-call accuracy.

The deterministic split logic lives in `split.py`, the single source of truth:
the eval holdout is carved out first, then train/val are split from the rest,
so the three sets are provably disjoint.

## Files

| File | Purpose |
|------|---------|
| `herdr_tools.py` | The 25 Herdr operations; schemas loaded from `reference/herdr_schemas.json`. |
| `split.py` | Single source of truth for the train/val/eval split (holdout carved out first). |
| `make_dataset.py` | Generates `dataset.jsonl` (chat format + structured labels). |
| `train_lfm2.py` | LoRA SFT driver (run on Colab GPU; see below). |
| `eval_lfm2.py` | Holdout eval; `--base` for baseline. |
| `eval_pi.mjs` | Same holdout, scored through pi's `ModelRuntime` for any catalog model (deepseek flash, GLM 5.3 flash); records tokens + cost. |
| `pin_holdout.py` | Persists the eval holdout (keyed by query) so re-eval stays comparable as the dataset grows. |
| `validate_dataset.py` | Live-validates dataset labels against a real `herdr` server. |
| `adapters/lfm2_herdr_lora/` | **Current tuned adapter (v7).** Older runs are archived alongside (`_v1`, `_v3`, `_v4`, `_v6`). |
| `reference/` | Herdr tool schemas, captured CLI help (`cli_help/`), API schema, skill doc. |
| `scripts/` | Colab glue (`setup_lfm2_colab.py`, `fix_torchao.py`, `run_detached_*.py`), one-off probes (`probe_*`), and `make_eval_graph.py` (renders `docs/eval_comparison.*`). |
| `runs/` | Experiment artifacts and durable results: checkpoint tarballs, training logs, eval snapshots + per-version `eval_v*_summary.md`. |
| `NOTES.md` | Why we dropped Needle 2; how to recover that track. |
| `Makefile` | `make data` / `make eval` / `make validate`; `make train` prints the Colab recipe. |

The pipeline chain is: `make data` -> train on Colab -> unpack the dumped
checkpoint into `adapters/lfm2_herdr_lora/` -> `make eval`.

## Train on Google Colab

A T4 is enough for 350M (~15 min). Use the `colab` CLI (or just run
`make train` to print this recipe):

```sh
colab new -s NAME --gpu T4
colab exec -s NAME -f scripts/setup_lfm2_colab.py   # transformers>=4.55 peft datasets accelerate
colab exec -s NAME --timeout 400 -f scripts/fix_torchao.py   # torchao>=0.16 (peft 0.20 requires it)
colab upload -s NAME dataset.jsonl /content/dataset.jsonl
colab upload -s NAME train_lfm2.py /content/train_lfm2.py
colab upload -s NAME split.py /content/split.py     # train_lfm2.py imports it
colab upload -s NAME runs/eval_v5_holdout.json /content/eval_v5_holdout.json   # optional: pinned holdout
```

Optional but recommended — mirror the checkpoint to Google Drive so it
survives a VM reap: `colab drivemount -s NAME` (interactive OAuth once), then
queue `scripts/copy_to_drive.py` as a second `colab exec` (it waits for the
checkpoint tarball and copies it to `/content/drive/MyDrive/herdr/`, sha256
verified).

(The Colab scripts write the adapter flat at `/content/lfm2_herdr_lora`; after
reconstructing the dumped tarball locally, unpack it into `adapters/`.)

NEVER run training inside one blocking `colab exec` — an exec timeout or a dead
keep-alive daemon lets Colab idle-prune the VM and you lose the checkpoint.
Use `run_detached_dump.py`: nohup-detach the trainer, poll every 120 s, tick
keep-alive every 60 s, then tar+base64-dump the checkpoint to stdout on
completion (the VM can be reaped seconds after training finishes). Reconstruct
locally with the snippet in `NOTES.md` / the script's docstring. To train with
the pinned holdout (so the 98 eval rows never leak into training), pass
`--env HOLDOUT=/content/eval_v5_holdout.json` to that `colab exec`.

Current recipe (v6/v7): epochs 12, batch 1, grad-accum 8, lr 1e-4, LoRA
r=16 alpha=32 on `q/k/v` + `w1/w3/w2`. LFM2 naming gotcha: the MLP
projections are `w1`/`w3`/`w2`, NOT gate/up/down_proj (those match nothing),
and attention output is `out_proj` which is SHARED with Lfm2ShortConv — there
is **no `o_proj` at all**, so the old "o_proj" target silently trained q/k/v
only. Do NOT target `conv_in_proj`/`conv_out_proj` — PEFT routes them through
torchao and crashes.

## Evaluate

Score the adapter against the pinned, query-keyed holdout (98 rows) so results
stay comparable as the dataset grows:

```sh
.venv/bin/python eval_lfm2.py --adapter adapters/lfm2_herdr_lora --holdout runs/eval_v5_holdout.json
.venv/bin/python eval_lfm2.py --base --holdout runs/eval_v5_holdout.json   # baseline
```

The holdout is pinned once via `pin_holdout.py` (keyed by query string, so
appending training rows never shifts it) and then reused for both eval and
train (`--holdout runs/eval_v5_holdout.json`, or `--env HOLDOUT=...` on Colab).

**Current result (v7)** — 98-row holdout, seed 42, strictly disjoint from
training, all 25 tools represented:

| model | exact-call | exact-norm | tool-selection | off-topic |
|---|---:|---:|---:|---:|
| base (untuned) | 9.8% | 9.8% | 26.8% | 68.8% |
| **v7 (current adapter)** | **96.3% (79/82)** | **96.3%** | **96.3%** | **100% (16/16)** |
| deepseek-v4-flash-vision-exp (pi harness) | 56.1% | 56.1% | 65.9% | 50.0% |
| glm-5.3-flash / OpenRouter (pi harness) | 73.2% | 76.8% | 87.8% | 50.0% |

> **Frontier-model comparison** (`eval_pi.mjs`): deepseek-v4-flash-vision-exp
> and glm-5.3-flash are scored through pi's `ModelRuntime` with the same 25
> Herdr tools and the same pinned holdout. Both trail the fine-tune badly on
> exact-call (56.1% and 73.2% vs 96.3%) and both act on half the off-topic
> prompts; the 98-row runs cost **$0.0069** (deepseek) and **$0.0080** (GLM).
> Full breakdowns: `runs/eval_deepseek_summary.md`, `runs/eval_glm_summary.md`.

> **v7 is a regime change — do not compare it to v6.** Through v6 every
> training/eval row shared ONE fixed system prompt (workspace=`w1`, pane=
> `w1:p1`, cwd=`/home/repo`, agent kind=hermes), which let the model "solve"
> grounding by memorizing constants. v7 deterministically rotates the system
> prompt over 8 contexts (workspaces w1–w5, cwds, caller panes, agent kinds),
> so it tests context-free grounding. It came out **+2.4 pts** over v6 anyway.
> The pinned holdout is keyed by query string only, so it still resolves; the
> per-run numbers and remaining failure modes are in `runs/eval_v7_summary.md`.

> **All pre-fix numbers (v1–v4, and any run before commit `798b7d8`) are
> invalid.** Eval's `ask()` used to render `row["messages"]` (including the
> gold assistant answer) with `add_generation_prompt=True`, scoring a
> *continuation* rather than a from-scratch answer. Fixed to
> `row["messages"][:-1]` in `eval_lfm2.py` to match `train_lfm2.py`. Re-run
> any adapter with the fixed `eval_lfm2.py` before comparing.

Runtime invariant: `pane_split` without explicit pane/current targets the
caller's pane; normalization makes it explicit (`current: true`). Eval reports
both raw and normalized accuracy.

## Safety notes

- Eval is side-effect free; it only compares predicted calls to labels.
- Do not close workspaces/tabs/panes/sessions you did not create, and never
  `herdr server stop` from an active session unless intended.
- Command syntax follows the installed `herdr` CLI (v0.8.2), captured under
  `cli_help/` — the installed CLI is the source of truth.

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and
redistribute for any purpose, including commercial use and distillation. See
the `LICENSE` file for the full text.
