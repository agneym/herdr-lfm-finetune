# Hermes ↔ Needle 2: a fine-tuned Herdr expert

Fine-tune a **Needle 2** (45M) model to be an expert on the **Herdr** terminal
multiplexer, then have the main agent (Hermes) ask it in natural language for
the right Herdr operation and run the result.

The pipeline, files, and both training paths (external GPU + local CPU) are
described below.

---

## How it works

The Herdr operations are declared as Needle tool schemas (OpenAI-style function
schemas).  The model is LoRA-tuned on JSONL examples of the form

```
{ "system":   "HERDR_ENV=1\nworkspace=w1\ntab=w1:t1\npane=w1:p1\ncwd=/home/repo\n...",
  "tools":    [ ...25 herdr tool schemas... ],   # same list every example
  "query":    "split my pane to the right in the current directory",
  "reasoning":"current=caller pane; direction=right; cwd=current dir",
  "answers":  [ {"name": "pane_split", "arguments": {"current": true, "direction": "right"}} ] }
```

Because Needle constrains every response to a grammar built from the schemas,
the tuned model emits **structured tool calls** whose arguments are grounded in
the request.  Hermes reads the returned operation and executes the `herdr`
command itself.

---

## Files

| File | Purpose |
|------|---------|
| `herdr_tools.py` | 25 Herdr ops as `@needle.tool` functions + the ground-truth `SCHEMAS`. Runtime executor (set `NEEDLE_HERDR_EXECUTE=1` to actually run herdr). |
| `make_dataset.py` | Generates `data.jsonl` (curated + templated examples, ~1-in-8 off-topic). |
| `data.jsonl` | The training set (225 examples, all 25 tools used, 12% off-topic). |
| `train_herdr_agent.py` | Portable training driver (auto batch-size: 2 on GPU, 1 on CPU). |
| `colab_herdr_finetune.ipynb` | One-click Google Colab notebook (GPU) to train the adapter. |
| `ask_herdr.py` | The query harness Hermes calls: turns a natural-language request into `herdr` operation(s). |
| `eval_model.py` | Scores a tuned (or base) model against held-out dataset labels. |
| `validate_dataset.py` | Live-validates the dataset's labels against a real `herdr` server. |
| `tuned.cact` | (build output) the tuned archive you load into `needle.Needle(weights=...)`. |

`tuned.cact` is not present yet — it is produced by the train + build steps
below.

---

## Option A — train on a GPU (recommended, full breadth)

Training the full 25-tool catalogue needs `max_len 4096` (the catalogue alone is
~2400 tokens), which is far too slow to run on a 7.8GB CPU-only box
(~105 s/step → many hours).  A GPU makes it a ~15–30 minute job.

**Use Google Colab (free tier / T4)** — it is the least-effort external service
with a real GPU and no account gate for this workload.

1. Open `colab_herdr_finetune.ipynb` in <https://colab.research.google.com>.
   - Either `File > Upload notebook` and pick `colab_herdr_finetune.ipynb`, or
     drag the file onto `https://colab.research.google.com`.
2. Runtime > Change runtime type > **GPU** (T4).  Run all cells in order.
3. When prompted, upload `data.jsonl`.
4. At the end, download `adapter.pkl` (a few MB) to this project folder.
5. Build the tuned archive on this box:

   ```sh
   .venv/bin/needle build checkpoints/needle2.pkl --lora adapter.pkl --out tuned.cact
   ```

*(Hugging Face works too but is clunkier: create a Space with the paid T4
hardware upgrader, or run this in a GPU Docker container. Colab is simpler. Any
CUDA machine works — just `pip install "cactus-needle[gpu]"` and run
`train_herdr_agent.py data.jsonl`.)*

---

## Option B — train locally on CPU (focused only)

Fit only a focused core subset (e.g. 6–8 ops) so examples fit `max_len ~1024`,
where per-step cost is tractable (~30–60 min).  Trade breadth for speed; keep the
low-level `herdr_tools.py` catalogue but trim the training `tools` list.

```sh
.venv/bin/needle finetune data_focused.jsonl --epochs 12 --batch-size 1 --max-len 1024 --out adapter.pkl
.venv/bin/needle build checkpoints/needle2.pkl --lora adapter.pkl --out tuned.cact
```

---

## Evaluate the tuned model

`eval_model.py` runs the model in planner mode over dataset queries and compares
the emitted tool calls to the ground-truth labels (no side effects). Metrics:
**exact-call accuracy** (tool name + every argument, order-insensitive),
**tool-selection accuracy**, per-tool argument grounding, and off-topic
behaviour (rows with `answers: []` must not call a tool). Mismatches print with
expected vs predicted so weak tools are easy to spot.

```sh
# baseline — the UNTUNED model, same queries (separate process: the engine
# holds one set of weights per process)
.venv/bin/python eval_model.py --base --split 0.15

# the real eval — the tuned model on a held-out slice
.venv/bin/python eval_model.py --weights tuned.cact --split 0.15
```

The holdout split is deterministic (seed 42) and its row indices are printed.
Hold exactly those rows out of training, or the numbers are optimistic:

```sh
.venv/bin/python eval_model.py --weights tuned.cact --split 0.15 \
    --save-train data_train.jsonl     # data.jsonl minus the holdout
# train on data_train.jsonl instead of data.jsonl, then eval with the same
# --split 0.15 --seed 42
```

Reference numbers (base model, 12-row slice, seed 42): exact-call 22.2%,
tool-selection 33.3%, off-topic 66.7% — the tuned model should clear these.
Expect noise on a 12-row slice; run the full 34-row holdout
(no `--limit`) for trustworthy numbers, and re-run each config a couple of
times (the sampler is unseeded) to gauge variance.

A second, optional layer: `--save-preds preds.jsonl` dumps every prediction;
feeding those predicted calls through the same live-herdr replay that
`validate_dataset.py` uses catches calls that are valid-shaped but reference
nonexistent panes/workspaces.

---

## Query the agent

```sh
# Planner: returns the herdr operation(s) for Hermes to run (no side effects)
.venv/bin/python ask_herdr.py --query "split my pane to the right and run the linter"

# Full loop: the tuned agent executes the op itself (needs a GPU/CPU build + live herdr)
NEEDLE_HERDR_EXECUTE=1 .venv/bin/python ask_herdr.py --run --query "wait for 'ok' in w4:p1"

# A herdr CLI command can also execute the returned command directly:
herdr pane split --current --direction right
```

Example output (planner mode):
```json
{ "type": "call",
  "operations": [ { "name": "pane_run",
                    "command": "herdr pane run w1:p1 cargo test",
                    "arguments": { "pane": "w1:p1", "command": "cargo test" } } ] }
```

---

## Wiring into Hermes (the main agent)

Inside a Herdr pane (so `HERDR_ENV=1` and the session IDs are set), the main
agent calls the tuned model, then runs the returned command:

1. Ask the expert:
   `OP=$(.venv/bin/python ask_herdr.py --query "READ THE QUERY HERE")`
2. Parse `operations[*].command` and execute it (or run all sequentially).
3. Feed results back by asking again with the next request (the agent holds no
   cross-turn memory by default; the planner is stateless).

The `ask_herdr.py` process is intentionally separate — Needle 2's engine can only
hold one set of weights per process, so the tuned model runs in its own process.

System facts the model grounds against (`current pane`, `cwd`) come from the live
`HERDR_WORKSPACE_ID` / `HERDR_TAB_ID` / `HERDR_PANE_ID` / `PWD` env vars, which
match the `system` block used during training.

---

## Safety notes

- The planner mode never calls herdr — it only returns commands.  Only the
  `--run` mode (and `NEEDLE_HERDR_EXECUTE=1`) executes them.
- Do not close workspaces/tabs/panes/sessions you did not create, and never
  `herdr server stop` from an active session unless explicitly intended.
- The dataset was validated against the local `herdr` binary (v0.8.2) CLI help;
  command syntax follows the installed CLI, which is the source of truth.
