"""Queued Drive backup for the detached lfm2v8 trainer.

Run AFTER training completes (this exec queues behind watch_and_dump.py). Copies
the checkpoint tarball the watcher already made (/content/ckpt.tar.gz) to Google
Drive, with size + sha256 verification. Polls briefly for the tar in case it is
still being written.
"""
import subprocess, time, os, shutil, hashlib

DRIVE_DIR = "/content/drive/MyDrive/herdr"
DRIVE_FILE = "ckpt-lfm2v8.tar.gz"
SRC = "/content/ckpt.tar.gz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Wait for the tarball (watch_and_dump.py creates it on TRAINING OK). 8h cap.
deadline = time.time() + 8 * 3600
while time.time() < deadline:
    if os.path.exists(SRC):
        break
    print(f"[wait] {SRC} not yet present", flush=True)
    time.sleep(60)
else:
    print("TIMEOUT waiting for checkpoint tarball", flush=True)
    raise SystemExit(1)

size = os.path.getsize(SRC)
h = sha256(SRC)
print(f"found {SRC}: {size} bytes sha256={h}", flush=True)

if not os.path.isdir("/content/drive/MyDrive"):
    print("Drive not mounted - aborting Drive copy", flush=True)
    raise SystemExit(1)

os.makedirs(DRIVE_DIR, exist_ok=True)
dest = os.path.join(DRIVE_DIR, DRIVE_FILE)
for attempt in range(3):
    shutil.copy2(SRC, dest)
    ok = (os.path.exists(dest) and os.path.getsize(dest) == size
          and sha256(dest) == h)
    print(f"copy attempt {attempt + 1}: {'OK' if ok else 'mismatch'}", flush=True)
    if ok:
        print(f"=== DRIVE COPY OK: {dest} ===", flush=True)
        break
    time.sleep(5)
else:
    print("DRIVE COPY FAILED after 3 attempts", flush=True)
