#!/usr/bin/env python3
"""Summarize PHASETIME/LENDIST records from a Gemma4 benchmark head log."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics


PHASE_RE = re.compile(r"PHASETIME step=(-?\d+)\s+(.*)")
PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=(-?\d+(?:\.\d+)?)")
TP_RE = re.compile(r"['\"]tensor_parallel_size['\"]:\s*(\d+)")
DP_RE = re.compile(r"['\"]data_parallel_size['\"]:\s*(\d+)")
LENDIST_RE = re.compile(
    r"LENDIST step=(-?\d+).*?truncated=(\d+) \(([0-9.]+)%\).*?"
    r"osl_mean=([0-9.]+).*?isl_mean=([0-9.]+)"
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-tp", required=True, type=int)
    parser.add_argument("--expected-dp", required=True, type=int)
    parser.add_argument("--expected-steps", required=True, type=int)
    args = parser.parse_args()

    phases: dict[int, dict[str, float]] = {}
    lengths: dict[int, dict[str, float]] = {}
    patch_contract = False
    failures: list[str] = []
    observed_mesh: tuple[int, int] | None = None
    clean_exit = False

    for line in args.log.read_text(errors="replace").splitlines():
        if "PATCH-CONTRACT: APPLIED" in line:
            patch_contract = True
        if "PATCH-CONTRACT: FAILED" in line:
            failures.append(line.strip())
        if "MaxText Job Ended" in line and "exit code 0" in line:
            clean_exit = True
        if "non-default args:" in line:
            tp_match = TP_RE.search(line)
            dp_match = DP_RE.search(line)
            observed_mesh = (
                int(tp_match.group(1)) if tp_match else 1,
                int(dp_match.group(1)) if dp_match else 1,
            )
        phase_match = PHASE_RE.search(line)
        if phase_match:
            step = int(phase_match.group(1))
            phases[step] = {
                key: float(value) for key, value in PAIR_RE.findall(phase_match.group(2))
            }
        length_match = LENDIST_RE.search(line)
        if length_match:
            step = int(length_match.group(1))
            lengths[step] = {
                "truncated": int(length_match.group(2)),
                "truncation_pct": float(length_match.group(3)),
                "osl_mean": float(length_match.group(4)),
                "isl_mean": float(length_match.group(5)),
            }

    if failures or not patch_contract:
        raise SystemExit(
            "RESULT: FAILED patch contract was not proven: "
            f"applied={patch_contract} failures={failures}"
        )
    if not clean_exit:
        raise SystemExit("RESULT: FAILED MaxText exit-code-zero marker is missing")
    expected_step_ids = set(range(args.expected_steps))
    if set(phases) != expected_step_ids:
        raise SystemExit(
            "RESULT: FAILED incomplete PHASETIME steps: "
            f"expected={sorted(expected_step_ids)} observed={sorted(phases)}"
        )
    if set(lengths) != expected_step_ids:
        raise SystemExit(
            "RESULT: FAILED incomplete LENDIST steps: "
            f"expected={sorted(expected_step_ids)} observed={sorted(lengths)}"
        )
    required_metrics = {"engine_generate", "compute_rewards", "run_global_step"}
    missing_metrics = {
        step: sorted(required_metrics - set(record))
        for step, record in phases.items()
        if not required_metrics.issubset(record)
    }
    if missing_metrics:
        raise SystemExit(f"RESULT: FAILED PHASETIME metrics missing: {missing_metrics}")
    warm = {step: record for step, record in phases.items() if step >= 1}
    expected_mesh = (args.expected_tp, args.expected_dp)
    if observed_mesh != expected_mesh:
        raise SystemExit(
            "RESULT: FAILED rollout mesh mismatch: "
            f"expected tp{expected_mesh[0]}xdp{expected_mesh[1]}, observed={observed_mesh}"
        )

    available_metrics = sorted({key for record in warm.values() for key in record})
    metric_stats: dict[str, dict[str, float]] = {}
    tails: dict[str, list[dict[str, float]]] = {}
    for metric in available_metrics:
        values = [record[metric] for record in warm.values() if metric in record]
        if not values:
            continue
        metric_stats[metric] = stats(values)
        median = metric_stats[metric]["median"]
        threshold = max(median * 1.5, median + 10.0)
        tails[metric] = [
            {"step": step, "seconds": record[metric]}
            for step, record in sorted(warm.items())
            if metric in record and record[metric] > threshold
        ]

    length_warm = [record for step, record in lengths.items() if step >= 1]
    length_summary = {}
    for metric in ("truncation_pct", "osl_mean", "isl_mean"):
        values = [float(record[metric]) for record in length_warm]
        if values:
            length_summary[metric] = stats(values)

    result = {
        "run": args.run,
        "profile": args.profile,
        "artifact_uri": f"gs://ubench-logs/alisachen-tpu7x/runs/{args.run}/",
        "post_cold_steps": sorted(warm),
        "rollout_mesh": {"tp": observed_mesh[0], "dp": observed_mesh[1]},
        "patch_contract_applied": patch_contract,
        "clean_exit": clean_exit,
        "failures": failures,
        "metrics_seconds": metric_stats,
        "tail_candidates": {key: value for key, value in tails.items() if value},
        "length_distribution": length_summary,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(
        f"RESULT run={args.run} profile={args.profile} "
        f"mesh=tp{observed_mesh[0]}xdp{observed_mesh[1]} steps={len(warm)}"
    )
    print(f"RESULT artifact={result['artifact_uri']}")
    print(f"RESULT patch_contract={'PASS' if patch_contract and not failures else 'FAIL'}")
    print("metric                         median      mean       p90       max")
    for metric in (
        "engine_generate",
        "compute_rewards",
        "generate_and_advantage",
        "micro_batch_steps",
        "run_global_step",
    ):
        if metric not in metric_stats:
            continue
        item = metric_stats[metric]
        print(
            f"{metric:28} {item['median']:9.2f} {item['mean']:9.2f} "
            f"{item['p90']:9.2f} {item['max']:9.2f}"
        )
    if result["tail_candidates"]:
        print("TAIL-CANDIDATES " + json.dumps(result["tail_candidates"], sort_keys=True))
    if length_summary:
        print(
            "VALIDITY "
            f"truncation_median={length_summary.get('truncation_pct', {}).get('median', float('nan')):.2f}% "
            f"osl_median={length_summary.get('osl_mean', {}).get('median', float('nan')):.1f} "
            f"isl_median={length_summary.get('isl_mean', {}).get('median', float('nan')):.1f}"
        )
    print(f"RESULT json={args.output}")


if __name__ == "__main__":
    main()
