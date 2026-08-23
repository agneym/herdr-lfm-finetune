import subprocess, threading, time

# keep-alive thread: run a trivial kernel call every 60s so Colab doesn't idle-prune,
# while nohup trains in the background.
stop = False
def keepalive():
    i = 0
    while not stop:
        time.sleep(60)
        i += 1
        print(f'[keepalive tick {i}]', flush=True)

t = threading.Thread(target=keepalive, daemon=True)
t.start()

subprocess.run("nohup python train_lfm2.py --data dataset.jsonl "
               "--epochs 8 --batch-size 1 --grad-accum 8 --lr 1e-4 "
               "--out lfm2_herdr_lora > train.log 2>&1 &", shell=True)
print('launched detached; polling train.log', flush=True)

# poll until the log says done or the process disappears
import os
while True:
    time.sleep(120)
    try:
        log = open('train.log').read()
        tail = log[-400:]
        alive = subprocess.run(['pgrep', '-f', 'train_lfm2.py'],
                               capture_output=True, text=True).stdout.strip()
        print(f'--- poll: proc={"UP" if alive else "DOWN"} ---\n{tail}', flush=True)
        if 'TRAINING OK' in log or 'TRAINING FAILED' in log or not alive:
            break
    except FileNotFoundError:
        print('no log yet', flush=True)
print('DONE WATCHING', flush=True)
