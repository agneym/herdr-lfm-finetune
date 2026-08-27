import argparse
import subprocess, threading, time, base64, os

stop = False
def keepalive():
    i = 0
    while not stop:
        time.sleep(60)
        i += 1
        print(f'[keepalive tick {i}]', flush=True)

t = threading.Thread(target=keepalive, daemon=True)
t.start()

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", default=None,
                help="pinned holdout JSON path on the VM (e.g. /content/eval_v5_holdout.json); "
                     "also read from the HOLDOUT env var (colab exec --env HOLDOUT=...).")
args = ap.parse_args()
holdout = args.holdout or os.environ.get("HOLDOUT")

cmd = ("rm -rf lfm2_herdr_lora; nohup python train_lfm2.py --data dataset.jsonl "
       "--epochs 12 --batch-size 1 --grad-accum 8 --lr 1e-4 "
       "--out lfm2_herdr_lora")
if holdout:
    cmd += f" --holdout {holdout}"
cmd += " > train.log 2>&1 &"
subprocess.run(cmd, shell=True)
print('launched detached; polling train.log', flush=True)

while True:
    time.sleep(120)
    try:
        log = open('train.log').read()
    except FileNotFoundError:
        print('no log yet', flush=True)
        continue
    alive = subprocess.run(['pgrep', '-f', 'train_lfm2.py'],
                           capture_output=True, text=True).stdout.strip()
    print(f'--- poll: proc={"UP" if alive else "DOWN"} ---\n{log[-300:]}', flush=True)
    if 'TRAINING OK' in log or 'TRAINING FAILED' in log or not alive:
        break

# IMMEDIATELY dump the checkpoint as base64 chunks to stdout so it lands in OUR log
print('=== CKPT DUMP START ===', flush=True)
r = subprocess.run(['tar', 'czf', 'ckpt.tar.gz', 'lfm2_herdr_lora'])
data = open('ckpt.tar.gz', 'rb').read()
b64 = base64.b64encode(data).decode()
CH = 60000
for i in range(0, len(b64), CH):
    print(b64[i:i+CH], flush=True)
print('=== CKPT DUMP END ===', flush=True)
print(f'dumped {len(data)} bytes', flush=True)
