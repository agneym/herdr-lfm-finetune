"""Watch-only companion to run_detached_dump.py.

Use when training was already launched detached (nohup) by a previous exec
whose wrapper timed out. Does NOT relaunch the trainer: polls train.log with
keep-alive ticks, and on TRAINING OK runs the identical tar+base64 dump so
the checkpoint lands in this exec's stdout log.
"""
import subprocess, time, base64

stop = False
def keepalive():
    i = 0
    while not stop:
        time.sleep(60)
        i += 1
        print(f'[keepalive tick {i}]', flush=True)

import threading
t = threading.Thread(target=keepalive, daemon=True)
t.start()

print('watching train.log (no relaunch)', flush=True)
while True:
    time.sleep(120)
    try:
        log = open('train.log').read()
    except FileNotFoundError:
        print('no train.log yet', flush=True)
        continue
    alive = subprocess.run(['pgrep', '-f', 'train_lfm2.py'],
                           capture_output=True, text=True).stdout.strip()
    print(f'--- poll: proc={"UP" if alive else "DOWN"} ---\n{log[-300:]}', flush=True)
    if 'TRAINING OK' in log or 'TRAINING FAILED' in log or not alive:
        break

if 'TRAINING OK' in log:
    print('=== CKPT DUMP START ===', flush=True)
    subprocess.run(['tar', 'czf', 'ckpt.tar.gz', 'lfm2_herdr_lora'])
    data = open('ckpt.tar.gz', 'rb').read()
    b64 = base64.b64encode(data).decode()
    CH = 60000
    for i in range(0, len(b64), CH):
        print(b64[i:i+CH], flush=True)
    print('=== CKPT DUMP END ===', flush=True)
    print(f'dumped {len(data)} bytes', flush=True)
else:
    print('TRAINING NOT OK - no dump', flush=True)
