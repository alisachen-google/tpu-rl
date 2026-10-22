# SPDX-License-Identifier: Apache-2.0
"""Run Raiden host-store processes on a decode host using prefill manifests."""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import time
from pathlib import Path


def build_commands(manifests, *, replica, server_ip, budget, binary):
    if budget <= 0:
        raise ValueError("DRAM budget per store must be positive")
    commands, seen = [], set()
    for path in manifests:
        manifest = json.loads(Path(path).read_text())
        if manifest["version"] != 1:
            raise ValueError(f"Unknown manifest version in {path}")
        if replica not in manifest["replicas"]:
            raise ValueError(f"Replica {replica} is not a configured target in {path}")
        group = manifest["group"]
        if not group or group in seen:
            raise ValueError(f"Empty or duplicate pool group in {path}")
        seen.add(group)
        block_bytes = sum(manifest["block_bytes"])
        if block_bytes <= 0 or budget < block_bytes:
            raise ValueError(f"Budget cannot hold one KV page for {group}")
        commands.append(
            [
                binary,
                f'--job_name={manifest["job_name"]}',
                f"--job_replica_id={replica}",
                f'--data_name={manifest["data_name"]}',
                "--data_replica_idx=0",
                f"--store_server_ip={server_ip}",
                f'--global_registry_address={manifest["global_registry_address"]}',
                f"--kv_pool_group={group}",
                f"--dram_budget_bytes={budget}",
                "--evict_tier=1",
            ]
        )
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", nargs="+", required=True, type=Path)
    parser.add_argument("--replica", default="0")
    parser.add_argument("--store-server-ip", required=True)
    parser.add_argument("--dram-budget-bytes-per-store", type=int, required=True)
    parser.add_argument("--binary", default="kv_cache_host_store_node_main")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commands = build_commands(
        args.manifest,
        replica=args.replica,
        server_ip=args.store_server_ip,
        budget=args.dram_budget_bytes_per_store,
        binary=args.binary,
    )
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return

    def stop(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, stop)
    processes = []
    try:
        for command in commands:
            processes.append(subprocess.Popen(command))
        while True:
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Host store {process.pid} exited with {process.returncode}"
                    )
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
