PY = .venv/bin/python
ADAPTER = adapters/lfm2_herdr_lora

.PHONY: data train eval validate clean

## data — regenerate dataset.jsonl (chat format + structured labels)
data:
	$(PY) make_dataset.py

## train — print the Colab recipe (training runs on a Colab GPU, not locally)
train:
	@echo "Training runs on Google Colab — see README > 'Train on Google Colab':"
	@echo "  colab new -s NAME --gpu T4"
	@echo "  colab exec -s NAME -f scripts/setup_lfm2_colab.py"
	@echo "  colab exec -s NAME --timeout 400 -f scripts/fix_torchao.py"
	@echo "  colab upload -s NAME dataset.jsonl /content/dataset.jsonl"
	@echo "  colab upload -s NAME train_lfm2.py /content/train_lfm2.py"
	@echo "  colab upload -s NAME split.py /content/split.py"
	@echo "  colab exec -s NAME -f scripts/run_detached_dump.py   # detached run + ckpt dump"

## eval — score the adapter on the holdout split
eval:
	$(PY) eval_lfm2.py --adapter $(ADAPTER) --split 0.15 | tee runs/eval_latest.txt

eval-base:
	$(PY) eval_lfm2.py --base --split 0.15 | tee runs/eval_base.txt

## validate — live-check dataset labels against a running herdr server
validate:
	$(PY) validate_dataset.py

clean:
	rm -rf __pycache__
