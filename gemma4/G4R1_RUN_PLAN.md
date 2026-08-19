# g4r1 campaign — gemma4-e2b decode-heavy e2e on tpu7x 4x4x4 with the tianyu stack (2026-08-19)

Goal: the "open rung #1" of GEMMA4_PERF_REPORT.md — the real math-RL workload (256×8,
8192/4096, tmbs=1, math rewards, 30 steps) with the full tianyu-rl-14 stack activated
(5 runtime patches, continue-decode window 64, sharded logprobs tp=8, async_scheduling=False,
proxy XLA flags @ scoped_vmem 65536, compile cache). Comparator: g4r0a 490 s median cycle.

Driven from VM `ac-ml-perf` (see tpu-rl chart/ — full chart reconstructed 2026-08-19 from
the validated render; `helm template | kubectl apply`, namespaced K8s SA token).

| Step | What | Launcher / values | Gate |
|---|---|---|---|
| S1 | 2x2x1 smoke, MaxText route, stack on (2 steps, batch 8, tp 4) | `gemma4_e2b_r1_launcher.sh` + `values.yaml` + `values-r1-smoke.yaml` | 5× `PATCH: APPLIED`, rewards > 0, 2/2 steps, exit 0 |
| S2 | 2x2x1 smoke, NATIVE route (Gemma4ForCausalLM + weight_mapping/gemma4.py) + stack — first time the two are combined | `gemma4_e2b_r1_native_launcher.sh` + same values | as S1 + 0 unmapped keys, EOS-terminating completions |
| R1 | 4x4x4 MaxText route, 30 steps | `gemma4_e2b_r1_launcher.sh` + `values.yaml` + `values-4x4x4.yaml` + `values-dl.yaml` | ≥15 warm cycles; phasetime split; LENDIST; reward trend |
| R1n | 4x4x4 native route (if S2 passes) | `gemma4_e2b_r1_native_launcher.sh` + same values | same; compare sampler routes like-for-like |
| Opt | RPA tuple `R1_RPA=1,8704,1,2176`, tp 4 vs 8, tmbs 2 scout | env knobs on the same launcher | per-rung timing |

Launch recipe:
```bash
RUN=g4r1-$(date +%m%d%H%M)
helm template $RUN chart -f chart/values.yaml -f chart/values-4x4x4.yaml -f chart/values-dl.yaml \
  --set-file workload_launcher=gemma4/gemma4_e2b_r1_launcher.sh --set-file workload_config=chart/config.yml > renders/$RUN.yaml
kubectl apply -f renders/$RUN.yaml
```
Pre-flight (cluster-scoped, needs user creds): 4x4x4 pool present & idle; no foreign pathways head
on the cpu-np node (Fix #22); `kubectl get localqueues` still has `default`.

Known trap: do NOT enable the tunix profiler together with PHASED_PROFILING_DIR — g4dlv2
(2026-08-18 17:53Z) died at step 1 with `start_trace called while a trace is already being taken`.
