"""Train, then TAR the adapter to a single file and HOLD for download.

Why: `colab download` only fetches single FILES (not directories), and the VM
is idle-reaped seconds after the blocking exec ends. So we:
  1. nohup-detach the trainer (exec timeout can't kill it)
  2. poll train.log with keep-alive ticks (VM not reaped during training)
  3. on TRAINING OK, `tar czf /content/ckpt.tar.gz lfm2_herdr_lora` so there is
     ONE downloadable file
  4. print READY, then hold the exec alive (keep-alive ticks) so the operator
     can run `colab download -s NAME /content/ckpt.tar.gz ./ckpt.tar.gz`
     without the VM being reaped.
"""
import subprocess, time, threading, os, sys

DOWNLOAD_WINDOW_MINUTES = int(os.environ.get("DOWNLOAD_WINDOW_MINUTES", "30"))
stop = False


def keepalive(tag):
    i = 0
    while not stop:
        time.sleep(60)
        i += 1
        print(f"[{tag} keepalive tick {i}]", flush=True)


def launch():
    subprocess.run(
        "rm -rf lfm2_herdr_lora; nohup python train_lfm2.py --data dataset.jsonl "
        "--epochs 8 --batch-size 1 --grad-accum 8 --lr 1e-4 "
        "--out lfm2_herdr_lora > train.log 2>&1 &",
        shell=True)
    print("launched trainer detached; log -> train.log", flush=True)


def poll_until_done():
    while True:
        time.sleep(60)
        try:
            log = open("train.log").read()
        except FileNotFoundError:
            print("[poll] no log yet", flush=True)
            continue
        alive = subprocess.run(["pgrep", "-f", "train_lfm2.py"],
                               capture_output=True, text=True).stdout.strip()
        status = "UP" if alive else "DOWN"
        tail = log[-300:].replace("\n", "\\n")
        print(f"[poll] proc={status}  {tail}", flush=True)
        if "TRAINING OK" in log or "TRAINING FAILED" in log or not alive:
            return log


def main():
    launch()
    t = threading.Thread(target=keepalive, args=("train",), daemon=True)
    t.start()
    log = poll_until_done()

    # Make the checkpoint a single downloadable file.
    r = subprocess.run("cd /content && tar czf ckpt.tar.gz lfm2_herdr_lora && "
                       "ls -la ckpt.tar.gz", shell=True, capture_output=True,
                       text=True)
    print("tar output:", flush=True)
    print(r.stdout or r.stderr, flush=True)
    if not os.path.exists("/content/ckpt.tar.gz"):
        print("TAR FAILED — aborting hold", flush=True)
        return

    print("=== READY FOR DOWNLOAD ===", flush=True)
    print("Run: colab download -s <SESSION> /content/ckpt.tar.gz ./ckpt.tar.gz", flush=True)
    stop = True
    t2 = threading.Thread(target=keepalive, args=("hold",), daemon=True)
    t2.start()
    deadline = time.time() + DOWNLOAD_WINDOW_MINUTES * 60
    while time.time() < deadline:
        time.sleep(60)
    print("=== DOWNLOAD WINDOW CLOSED ===", flush=True)


if __name__ == "__main__":
    main()
