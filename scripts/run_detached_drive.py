"""Train detached, tar, then auto-copy to Google Drive.

Why Drive: colab download via Contents API base64-encodes large ckpt.tar.gz into
JSON and 404s on ~20-40MB; Drive copy survives VM reap.

Flow (all inside one colab exec so keep-alive daemon stays):
  1. nohup-detach train_lfm2.py -> train.log
  2. poll every 60s + keepalive tick (so kernel not idle)
  3. on TRAINING OK: tar czf /content/ckpt.tar.gz lfm2_herdr_lora
  4. copy to /content/drive/MyDrive/herdr/ckpt-lfm2v6.tar.gz (drive must be
     mounted beforehand via `colab drivemount -s NAME`; this script verifies
     the mount and waits for it if missing)
  5. verify copy (size + sha256), also copy train.log
  6. hold briefly for optional colab download fallback

Usage:
  colab new -s NAME --gpu T4
  colab drivemount -s NAME              # mount to /content/drive, auth once
  colab exec -s NAME --timeout 7200 -f scripts/run_detached_drive.py

Env overrides:
  DRIVE_DIR=/content/drive/MyDrive/herdr  DOWNLOAD_WINDOW_MINUTES=15
"""
import subprocess, time, threading, os, sys, shutil, hashlib

DRIVE_DIR = os.environ.get("DRIVE_DIR", "/content/drive/MyDrive/herdr")
DRIVE_FILE = os.environ.get("DRIVE_FILE", "ckpt-lfm2v6.tar.gz")
DOWNLOAD_WINDOW_MINUTES = int(os.environ.get("DOWNLOAD_WINDOW_MINUTES", "15"))
TRAIN_CMD = (
    "rm -rf lfm2_herdr_lora; nohup python train_lfm2.py --data dataset.jsonl "
    "--epochs 8 --batch-size 1 --grad-accum 8 --lr 1e-4 "
    "--out lfm2_herdr_lora > train.log 2>&1 &"
)

stop = False

def keepalive(tag):
    i = 0
    while not stop:
        time.sleep(60)
        i += 1
        print(f"[{tag} keepalive tick {i}]", flush=True)

def launch():
    subprocess.run(TRAIN_CMD, shell=True)
    print("launched trainer detached; log -> train.log", flush=True)

def poll_until_done():
    while True:
        time.sleep(60)
        try:
            log = open("train.log").read()
        except FileNotFoundError:
            print("[poll] no log yet", flush=True)
            continue
        alive = subprocess.run(["pgrep", "-f", "train_lfm2.py"], capture_output=True, text=True).stdout.strip()
        status = "UP" if alive else "DOWN"
        tail = log[-400:].replace("\n", "\\n")
        print(f"[poll] proc={status}  {tail}", flush=True)
        if "TRAINING OK" in log or "TRAINING FAILED" in log or not alive:
            return log

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    global stop
    launch()
    t = threading.Thread(target=keepalive, args=("train",), daemon=True)
    t.start()
    log = poll_until_done()

    # tar
    print("=== TARRING ===", flush=True)
    r = subprocess.run("cd /content && tar czf ckpt.tar.gz lfm2_herdr_lora && ls -lh ckpt.tar.gz && sha256sum ckpt.tar.gz", shell=True, capture_output=True, text=True)
    print(r.stdout or r.stderr, flush=True)
    if not os.path.exists("/content/ckpt.tar.gz"):
        print("TAR FAILED — aborting", flush=True)
        return
    src_size = os.path.getsize("/content/ckpt.tar.gz")
    src_hash = sha256("/content/ckpt.tar.gz")
    print(f"src ckpt.tar.gz {src_size} bytes sha256={src_hash}", flush=True)

    # ensure drive mounted — if not, try mount (this will hang waiting for auth if not
    # pre-mounted via `colab drivemount`; the outer exec's 600s timeout covers the
    # OAuth browser step). We prefer pre-mounted, so just warn and wait briefly.
    print(f"=== DRIVE CHECK {DRIVE_DIR} ===", flush=True)
    drive_ok = os.path.isdir("/content/drive/MyDrive")
    print(f"drive mounted: {drive_ok}", flush=True)
    if not drive_ok:
        print("Drive not mounted. Attempting drive.mount (requires browser auth)...", flush=True)
        try:
            subprocess.run([sys.executable, "-c", "from google.colab import drive; drive.mount('/content/drive')"], timeout=600)
        except Exception as e:
            print(f"drive.mount failed: {e}", flush=True)
        drive_ok = os.path.isdir("/content/drive/MyDrive")
        print(f"drive mounted after attempt: {drive_ok}", flush=True)
    if not drive_ok:
        print("DRIVE NOT MOUNTED — skipping Drive copy, holding for manual download only", flush=True)
    else:
        os.makedirs(DRIVE_DIR, exist_ok=True)
        dest = os.path.join(DRIVE_DIR, DRIVE_FILE)
        log_dest = os.path.join(DRIVE_DIR, "train-lfm2v6.log")
        print(f"=== COPYING to {dest} ===", flush=True)
        for attempt in range(3):
            try:
                shutil.copy2("/content/ckpt.tar.gz", dest)
                # also copy log
                try:
                    shutil.copy2("train.log", log_dest)
                except Exception as e:
                    print(f"log copy warn: {e}", flush=True)
                # verify
                if os.path.exists(dest):
                    dst_size = os.path.getsize(dest)
                    dst_hash = sha256(dest)
                    print(f"dest {dst_size} bytes sha256={dst_hash}", flush=True)
                    if dst_size == src_size and dst_hash == src_hash:
                        print(f"=== UPLOAD OK {dest} ===", flush=True)
                        print(f"Verify locally: Drive MyDrive/herdr/{DRIVE_FILE}", flush=True)
                        break
                    else:
                        print(f"mismatch size/hash, retry {attempt+1}/3", flush=True)
                else:
                    print(f"dest missing after copy, retry {attempt+1}/3", flush=True)
            except Exception as e:
                print(f"copy attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(5)
        else:
            print("DRIVE COPY FAILED after 3 attempts — holding for manual download", flush=True)

    # hold for optional colab download fallback
    print("=== READY FOR DOWNLOAD ===", flush=True)
    print("Run: colab download -s <SESSION> /content/ckpt.tar.gz ./ckpt.tar.gz", flush=True)
    print(f"Or pull from Drive: {DRIVE_DIR}/{DRIVE_FILE}", flush=True)
    stop = True
    t2 = threading.Thread(target=keepalive, args=("hold",), daemon=True)
    t2.start()
    deadline = time.time() + DOWNLOAD_WINDOW_MINUTES * 60
    while time.time() < deadline:
        time.sleep(60)
    print("=== DOWNLOAD WINDOW CLOSED ===", flush=True)

if __name__ == "__main__":
    main()
