import subprocess
cmd = ["python", "train_lfm2.py", "--data", "dataset.jsonl",
       "--epochs", "8", "--batch-size", "1", "--grad-accum", "8",
       "--lr", "1e-4", "--out", "lfm2_herdr_lora"]
r = subprocess.run(cmd)
print('exit code:', r.returncode, flush=True)
print('TRAINING OK' if r.returncode == 0 else 'TRAINING FAILED', flush=True)
