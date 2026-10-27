#!/mnt/disks/scratch/xpk-venv/bin/python3
"""Delete UNCOMMITTED orbax checkpoint step dirs (no commit_success.txt) under gs://ubench-logs/<run>/checkpoints/actor/.
A failed async save leaves ~100k objects; on the next save at that step orbax removes them SEQUENTIALLY on
GCS (hours) and the trainer hangs (seed 42, 2026-10-27 08:39). Batched parallel deletes take ~100 s.
Usage: clean-uncommitted-ckpts.py <run_name>"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
run = sys.argv[1]
client = storage.Client(project="gke-aishared-gsc-dev"); b = client.bucket("ubench-logs")
steps = {}
for x in b.list_blobs(prefix=f"{run}/checkpoints/actor/"):
    parts = x.name.split("/")
    if len(parts) >= 5 and parts[3].isdigit():
        steps.setdefault(int(parts[3]), set()).add(parts[4])
t0 = time.time()
for s in sorted(steps):
    if "commit_success.txt" in steps[s]:
        continue
    names = [x.name for x in b.list_blobs(prefix=f"{run}/checkpoints/actor/{s}/")]
    print(f"[clean] {run} step {s} uncommitted: deleting {len(names)} objects", flush=True)
    def delete_chunk(chunk):
        try:
            with client.batch(raise_exception=False):
                for n in chunk: b.blob(n).delete()
        except Exception:
            for n in chunk:
                try: b.blob(n).delete()
                except Exception: pass
        return len(chunk)
    chunks = [names[i:i + 100] for i in range(0, len(names), 100)]
    with ThreadPoolExecutor(max_workers=32) as ex: list(ex.map(delete_chunk, chunks))
    print(f"[clean] step {s} remaining {sum(1 for _ in b.list_blobs(prefix=f'{run}/checkpoints/actor/{s}/'))} ({time.time()-t0:.0f}s)", flush=True)
print(f"[clean] committed steps: {sorted(s for s in steps if 'commit_success.txt' in steps[s])}", flush=True)
