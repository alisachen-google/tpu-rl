#!/usr/bin/env python3
"""Fail-closed validation for a rendered Gemma4 benchmark JobSet."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROFILES = {
    "tp1-dp64": {
        "DL_TP": "1",
        "DL_DP": "-1",
        "DL_TFRAC": "0.5",
        "DL_SFRAC": "0.5",
        "DL_PADFIX": "0",
        "effective_dp": 64,
    },
    "tp2-dp32": {
        "DL_TP": "2",
        "DL_DP": "-1",
        "DL_TFRAC": "0.5",
        "DL_SFRAC": "0.5",
        "DL_PADFIX": "1",
        "effective_dp": 32,
    },
    "tp4-dp32": {
        "DL_TP": "4",
        "DL_DP": "-1",
        "DL_TFRAC": "1.0",
        "DL_SFRAC": "1.0",
        "DL_PADFIX": "1",
        "effective_dp": 32,
    },
}

COMMON_ENV = {
    "DL_TMBS": "8",
    "DL_PACK": "8192",
    "DL_STEPS": "20",
    "DL_BUCKET_GAP": "32768",
    "DL_CKPT": "False",
    "DL_KEEPKV": "1",
    "DL_MVPROCS": "64",
    "DL_NOEOS": "1",
    "DL_VTMO": "5",
    "SKIP_JAX_PRECOMPILE": "1",
}


def fail(message: str) -> None:
    raise SystemExit(f"RENDER-CONTRACT: FAILED {message}")


def one(items: list[dict], kind: str, name: str) -> dict:
    matches = [
        item
        for item in items
        if item.get("kind") == kind and item.get("metadata", {}).get("name") == name
    ]
    if len(matches) != 1:
        fail(f"expected one {kind}/{name}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("render", type=Path)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--run", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    try:
        documents = [doc for doc in yaml.safe_load_all(args.render.read_text()) if doc]
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot parse {args.render}: {exc}")

    jobset = one(documents, "JobSet", args.run)
    patch_cm = one(documents, "ConfigMap", f"{args.run}-patches")
    launcher_cm = one(documents, "ConfigMap", f"{args.run}-launcher")

    if jobset.get("metadata", {}).get("namespace") != args.namespace:
        fail("JobSet namespace mismatch")
    if patch_cm.get("metadata", {}).get("namespace") != args.namespace:
        fail("patch ConfigMap namespace mismatch")
    labels = jobset.get("metadata", {}).get("labels", {})
    if labels.get("kueue.x-k8s.io/queue-name") != args.queue:
        fail(f"queue mismatch: {labels}")
    if jobset.get("spec", {}).get("failurePolicy", {}).get("maxRestarts") != 5:
        fail("JobSet maxRestarts must be exactly 5")

    replicated = {
        item["name"]: item for item in jobset["spec"].get("replicatedJobs", [])
    }
    if set(replicated) != {"pathways-head", "worker"}:
        fail(f"unexpected replicated jobs: {sorted(replicated)}")

    head_pod = replicated["pathways-head"]["template"]["spec"]["template"]
    head_spec = head_pod["spec"]
    head = next(
        (c for c in head_spec.get("containers", []) if c.get("name") == "jax-tpu"),
        None,
    )
    if head is None:
        fail("jax-tpu head container missing")
    if head_spec.get("priorityClassName") != args.priority:
        fail(f"head priority mismatch: {head_spec.get('priorityClassName')!r}")
    if head.get("image") != args.image:
        fail(f"image mismatch: {head.get('image')!r}")

    env = {entry["name"]: str(entry.get("value", "")) for entry in head.get("env", [])}
    expected = dict(COMMON_ENV)
    expected.update({k: v for k, v in PROFILES[args.profile].items() if k != "effective_dp"})
    expected["BENCH_PROFILE"] = args.profile
    mismatches = {
        key: (value, env.get(key))
        for key, value in expected.items()
        if env.get(key) != value
    }
    if mismatches:
        fail(f"environment mismatch: {mismatches}")

    annotations = head_pod.get("metadata", {}).get("annotations", {})
    if annotations.get("cluster-autoscaler.kubernetes.io/safe-to-evict") != "false":
        fail("safe-to-evict annotation was lost")
    mounts = {mount.get("name") for mount in head.get("volumeMounts", [])}
    volumes = {volume.get("name"): volume for volume in head_spec.get("volumes", [])}
    if "workload-patches" not in mounts:
        fail("workload-patches is not mounted on the head")
    patch_volume = volumes.get("workload-patches", {})
    if patch_volume.get("configMap", {}).get("name") != f"{args.run}-patches":
        fail(f"workload-patches volume mismatch: {patch_volume}")

    workers = replicated["worker"]["template"]["spec"]
    worker_pod = workers["template"]["spec"]
    if worker_pod.get("priorityClassName") != args.priority:
        fail(f"worker priority mismatch: {worker_pod.get('priorityClassName')!r}")
    worker = next(
        (c for c in worker_pod.get("containers", []) if c.get("name") == "pathways-worker"),
        None,
    )
    if worker is None:
        fail("pathways-worker container missing")
    tpu_limit = worker.get("resources", {}).get("limits", {}).get("google.com/tpu")
    if int(workers.get("parallelism", 0)) != 16 or str(tpu_limit) != "4":
        fail(
            "expected 16 worker pods x 4 TPU chips, got "
            f"parallelism={workers.get('parallelism')} tpu_limit={tpu_limit}"
        )
    selector = worker_pod.get("nodeSelector", {})
    if selector.get("cloud.google.com/gke-tpu-topology") != "4x4x4":
        fail(f"topology mismatch: {selector}")

    manifest_paths = [
        (args.manifest.parent / line.split(maxsplit=1)[1]).resolve()
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    manifest_names = {path.name for path in manifest_paths}
    patch_names = set(patch_cm.get("data", {}))
    if patch_names != manifest_names:
        fail(
            "patch ConfigMap keys differ from checksum manifest: "
            f"missing={sorted(manifest_names - patch_names)} "
            f"extra={sorted(patch_names - manifest_names)}"
        )
    if len(patch_names) != 33:
        fail(f"expected 33 patch bundle files, found {len(patch_names)}")
    content_mismatches = [
        path.name
        for path in manifest_paths
        if not path.exists()
        or path.read_text().rstrip("\n")
        != str(patch_cm["data"][path.name]).rstrip("\n")
    ]
    if content_mismatches:
        fail(f"rendered patch contents differ: {content_mismatches}")

    launcher = launcher_cm.get("data", {}).get("launch-workload.sh", "")
    if "PATCH-CONTRACT: APPLIED" not in launcher or "PATCH_ORDER=(" not in launcher:
        fail("fail-closed benchmark launcher is not embedded")

    print(
        "RENDER-CONTRACT: PASSED "
        f"profile={args.profile} mesh=tp{env['DL_TP']}xdp{PROFILES[args.profile]['effective_dp']} "
        "chips=64 core_devices=128 patches=33 steps=20"
    )


if __name__ == "__main__":
    main()
