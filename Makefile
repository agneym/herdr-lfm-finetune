PY = .venv/bin/python
ADAPTER = adapters/lfm2_herdr_lora

.PHONY: data train eval validate check-ignore clean

## data — regenerate dataset.jsonl (chat format + structured labels)
data:
	$(PY) make_dataset.py

## pin-holdout — persist the current eval holdout (keyed by query string)
## NOTE: pin_holdout.py refuses to overwrite without --force. The live pin is
## now runs/eval_v8_holdout.json (120 rows); if the dataset grows again, create
## a NEW versioned file (v9) explicitly, then repoint eval/train + eval_pi.mjs.
pin-holdout:
	$(PY) pin_holdout.py --data dataset.jsonl --out runs/eval_v8_holdout.json

## train — print the Colab recipe (training runs on a Colab GPU, not locally)
train:
	@echo "Training runs on Google Colab — see README > 'Training (Google Colab)':"
	@echo "  colab new -s NAME --gpu T4"
	@echo "  colab exec -s NAME -f scripts/setup_lfm2_colab.py"
	@echo "  colab exec -s NAME --timeout 400 -f scripts/fix_torchao.py"
	@echo "  colab upload -s NAME dataset.jsonl /content/dataset.jsonl"
	@echo "  colab upload -s NAME train_lfm2.py /content/train_lfm2.py"
	@echo "  colab upload -s NAME split.py /content/split.py"
	@echo "  colab upload -s NAME runs/eval_v8_holdout.json /content/eval_v8_holdout.json   # optional: pinned holdout"
	@echo "  colab exec -s NAME -f scripts/run_detached_dump.py --env HOLDOUT=/content/eval_v8_holdout.json   # detached run + ckpt dump"

## eval — score the adapter on the holdout split
eval:
	$(PY) eval_lfm2.py --adapter $(ADAPTER) --holdout runs/eval_v8_holdout.json | tee runs/eval_latest.txt

eval-base:
	$(PY) eval_lfm2.py --base --holdout runs/eval_v8_holdout.json | tee runs/eval_base.txt

## validate — live-check dataset labels against a running herdr server
validate:
	$(PY) validate_dataset.py

## check-ignore — fail if any tracked file is being ignored (should print nothing)
check-ignore:
	@echo "tracked-but-ignored files (expected: none):"
	@git ls-files -ci --exclude-standard

clean:
	rm -rf __pycache__
