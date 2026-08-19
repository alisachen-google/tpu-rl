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

## 2026-08-19 campaign log (native route)

- Root-caused + fixed native-route gibberish at rollout_tensor_parallelism>1: fused qkv/gate_up must be
  interleaved per TP shard (tpu_inference merged-linear layout). `patches/patch_gemma4_tp_fusion.py`
  + `GEMMA4_NATIVE_TP`; maxtext fork `ironwood-rl-gemma4` @ 35b7903. Validated: g4n4s (tp=4) coherent,
  X2/g4nx2s (tp=1) coherent, g4n1s/g4nx1s (tp=4, old fusion) gibberish → reward 0, mean_length 4k cap.
- aysu serving tuning (batched RPA + block 256) measured at 2x2x1: regression (see tuning/AYSU_ROLLOUT_TUNING.md).
- PR #4409 needs tunix half: `patches/patch_tunix_chunked_logps.py` (image tunix 0.1.7-snapshot);
  smoke g4n9s (tmbs=4, chunk 1024) in progress.
- 4x4x4 native+stack g4r1n-08191021 (tp=4/dp16, priority high): admitted 10:41Z after 21 min queue,
  16 workers on pool gke-tpu-2d1077c3, all patches applied, Num_devices 64 — then EVICTED 10:48Z by
  Kueue cohort reclamation (preemptor /tpu-shared-cohort/ubench-regression-tests; we borrow from the
  cohort in ClusterQueue `default`). Requeued automatically. Ask: nominal 64-chip quota for `default`
  or access to a queue with nominal quota (priority-dev/multislice-queue).
- Kueue priority: ubench uses `high` (750); our `medium` (500) gets preempted within `default`. Use
  `--set kueue_priority_class=high` for 4x4x4 runs (does not protect against cohort reclamation).
