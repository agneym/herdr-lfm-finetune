"""Tar the trained adapter into a single file at /content/ckpt.tar.gz."""
import subprocess, os

for p in ["lfm2_herdr_lora", "/content/lfm2_herdr_lora"]:
    if os.path.isdir(p):
        print(f"found dir: {p}", flush=True)
        files = os.listdir(p)
        print("contents:", files, flush=True)

r = subprocess.run(
    "cd /content && tar czf ckpt.tar.gz lfm2_herdr_lora && ls -la ckpt.tar.gz",
    shell=True, capture_output=True, text=True)
print(r.stdout, flush=True)
print(r.stderr, flush=True)
