PY = .venv/bin/python
ADAPTER = adapters/lfm2_herdr_lora
HF_REPO ?= agney/lfm2-herdr-lora
GGUF_REPO ?= agney/lfm2-herdr-gguf
GGUF ?= runs/export/lfm2-herdr-Q4_K_M.gguf

.PHONY: data fetch train eval validate gguf gguf-push gguf-eval check-ignore clean

## data — regenerate dataset.jsonl (chat format + structured labels)
data:
	$(PY) make_dataset.py

## fetch — download the tuned adapter from Hugging Face Hub into adapters/
## (the trainer weights are not committed to git; this is what makes `make eval`
##  work on a fresh clone). Override the repo with HF_REPO=<user>/<repo>.
fetch:
	$(PY) scripts/fetch_adapter.py --repo $(HF_REPO) --out $(ADAPTER)

## pin-holdout — persist the current eval holdout (keyed by query string)
## NOTE: pin_holdout.py refuses to overwrite without --force. The live pin is
## now runs/results/eval_v8_holdout.json (120 rows); if the dataset grows again, create
## a NEW versioned file (v9) explicitly, then repoint eval/train + eval_pi.mjs.
pin-holdout:
	$(PY) pin_holdout.py --data dataset.jsonl --out runs/results/eval_v8_holdout.json

## train — print the Colab recipe (training runs on a Colab GPU, not locally)
train:
	@echo "Training runs on Google Colab — see README > 'Training (Google Colab)':"
	@echo "  colab new -s NAME --gpu T4"
	@echo "  colab exec -s NAME -f scripts/setup_lfm2_colab.py"
	@echo "  colab exec -s NAME --timeout 400 -f scripts/fix_torchao.py"
	@echo "  colab upload -s NAME dataset.jsonl /content/dataset.jsonl"
	@echo "  colab upload -s NAME train_lfm2.py /content/train_lfm2.py"
	@echo "  colab upload -s NAME split.py /content/split.py"
	@echo "  colab upload -s NAME runs/results/eval_v8_holdout.json /content/eval_v8_holdout.json   # optional: pinned holdout"
	@echo "  colab exec -s NAME -f scripts/run_detached_dump.py --env HOLDOUT=/content/eval_v8_holdout.json   # detached run + ckpt dump"

## eval — score the adapter on the holdout split
eval:
	$(PY) eval_lfm2.py --adapter $(ADAPTER) --holdout runs/results/eval_v8_holdout.json | tee runs/results/eval_latest.txt

eval-base:
	$(PY) eval_lfm2.py --base --holdout runs/results/eval_v8_holdout.json | tee runs/results/eval_base.txt

## validate — live-check dataset labels against a running herdr server
validate:
	$(PY) validate_dataset.py

## gguf — merge the adapter + export a GGUF for llama.cpp (no Python runtime)
## Requires a llama.cpp checkout with a built llama-quantize; point LLAMA_CPP_DIR
## at it (default /tmp/llama.cpp). See scripts/export_gguf.py for the build recipe.
gguf:
	$(PY) scripts/export_gguf.py --adapter $(ADAPTER)

## gguf-push — same, then upload the GGUFs (+ auto model card) to HF Hub
gguf-push:
	$(PY) scripts/export_gguf.py --adapter $(ADAPTER) --push --repo $(GGUF_REPO)

## gguf-eval — score a GGUF against the pinned holdout (needs llama-server built)
## Override the file with GGUF=...; default is the Q4_K_M export.
gguf-eval:
	$(PY) scripts/eval_gguf.py --gguf $(GGUF) --holdout runs/results/eval_v8_holdout.json --spawn

## check-ignore — fail if any tracked file is being ignored (should print nothing)
check-ignore:
	@echo "tracked-but-ignored files (expected: none):"
	@git ls-files -ci --exclude-standard

clean:
	rm -rf __pycache__
