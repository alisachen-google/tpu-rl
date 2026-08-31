# gemma4-E2B RL Benchmark — HANDOFF BOOK

> **G4R4 MESH CORRECTION (2026-08-31).** g4r4's log (`api_utils` line) shows `data_parallel_size: 16`,
> devices 64-127 (half slice): g4r4 was **tp4×dp16**, NOT tp4×dp32 as recorded below. Same mesh as
> g4r10f. Its good rollout vs g4r10f's collapse = corrupted-era short generations (light KV), not mesh.
> True tp4×dp32 first runs as g4r12r2 (full slice + DL_HBM=0.25 DL_TMBS=4). Also: full-slice serial
> OOMs the train step at default HBM (vLLM's 0.6 reservation leaves 36.7G < 80.3G needed; g4r12r1) --
> half-slice never hits this because trainer devices are disjoint from vLLM's.

**Updated:** 2026-08-31 · **Repo:** `tpu-rl` (github.com/alisachen-google/tpu-rl) · **Full history:** `gemma4/GEMMA4_PERF_REPORT.md`

Self-contained handoff for another agent to continue the gemma4-E2B GRPO rollout-perf
campaign on TPU v7x. Everything needed to launch, read results, and continue is here.

---

## 0. TL;DR — where we are

- **Old champion:** g4r7b, global **125.8 s** (tp=1×dp64, 20-step median).
- **Current best:** tp=2×dp32 + padfix, global median **125.4 s** (g4r11, 20-step) — **statistically TIED
  with the old champion**, NOT the −11 s the 6-step probe suggested (short probes overstate; the tail
  dominates). Rollout is marginally better (95 vs 97).
- **THE REAL PROBLEM IS THE TAILS.** Median ~125 but MEAN ~162. Two tail sources:
  1. **Reward spikes** (up to 252 s, from unbounded math_verify sympy stragglers) — biggest.
  2. **Rollout spikes** (up to 300 s, from KV-pool saturation waves).
- **Root cause of everything:** rollout is HOST-BOUND (device ~2-3% util). The single-core Python head
  drives N engines; `device_get`/`block_until_ready` waits through the Pathways proxy dominate. Verified
  on our runs AND tianyu-rl-148 (their `block_until_ready` = 73.5 s dominates their host trace too).
- **In flight:** g4r13 = tp=2 + fp8 KV + verify-timeout, 8-step (tests both tail fixes).

---

## 1. Launch recipe

- **Image:** `gcr.io/supercomputer-testing/maxtext-rl-ironwood:2026-08-24-packing` (tunix 408ca1d9)
- **Launcher:** `gemma4/gemma4_e2b_r1_launcher.sh` (staged copy `/mnt/disks/scratch/g4r8-stage/g4r8_launcher.sh`)
- **Chart:** `chart/` (helm; templates a patches-configmap)
- **Namespace/queue:** `ubench-regression-tests` (OWNED quota, 256 chips) via LocalQueue `multislice-queue`,
  priority `medium`(500). Access requires membership in `cmcs-ai-ninja-team@twosync.google.com` (granted).
- **Staging dir:** `/mnt/disks/scratch/g4r8-stage/` (chart + launcher + patches + env overlays)
- **Topology:** 4x4x4 (64 chips → 64 rollout devices; tp sets dp = 64/tp)

### Launch command (template)
```
export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)   # ALWAYS first
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap ; NS=ubench-regression-tests
STAGE=/mnt/disks/scratch/g4r8-stage
helm --kube-context=$CTX install <run> $STAGE -f $STAGE/values-base.yaml -f $STAGE/values-4x4x4.yaml -f $STAGE/<run>-envs.yaml \
  --namespace $NS --set namespace=$NS \
  --set-file workload_launcher=$STAGE/g4r8_launcher.sh \
  --set workload.image=gcr.io/supercomputer-testing/maxtext-rl-ironwood:2026-08-24-packing \
  --set workload_patches=true --set patches.enabled=true --set workload.configFile= \
  --set maxRestarts=50 --set queue=multislice-queue --set kueue_priority_class=medium
```
- `--namespace` AND `--set namespace=` BOTH needed (jobset uses Release.Namespace; configmaps use .Values.namespace).
- `--set maxRestarts=50` (default 0 wastes restarts on preemption).
- Reinstall only AFTER old pods fully terminate (a ~20 s race fails the jobset).
- Env overlays are `gemma4/g4rXX-envs.yaml` (a `workload.envs` list).

### Env knobs (DL_*), current champion-ish values
| Knob | Effect | value |
|---|---|---|
| DL_TP / DL_DP | rollout tensor / data parallelism (dp=-1 auto = 64/tp) | **2 / -1** (→dp32) |
| DL_TMBS / DL_PACK | train micro-batch / packing tokens | 8 / 8192 |
| DL_STEPS | num_batches (probe length) | 8-20 |
| DL_CKPT | enable_checkpointing (**False for probes** — avoids key<fry> restore crash) | False |
| DL_BUCKET_GAP | VLLM_TPU_BUCKET_PADDING_GAP — **CORRUPTION FIX, always on** | **32768** |
| DL_PADFIX | SS_PADFIX — activate vocab-sharded sampler at tp>1 | 1 (tp>1) |
| DL_KEEPKV | SYNC_KEEPKV keep KV across weight sync | 1 |
| DL_MVPROCS | math_verify_num_procs | 64 |
| DL_NOEOS | CD_NO_EOS_REDUCE drop per-iter cross-DP EOS reduce | 1 |
| DL_VTMO | VERIFY_ITEM_TIMEOUT — **per-item math_verify timeout (reward-tail fix)** | 5 |
| DL_ENGKW | RL_VLLM_ENGINE_KWARGS json (e.g. fp8 KV, prefix off) | varies |
| DL_JITU | JIT_UNROLL jitted weight-sync unroll | 0/1 |
| DL_ATTN/DL_SAQ/DL_SAKV | trainer attention / splash blocks | dot_product |
| DL_TFRAC | trainer_devices_fraction (Meta main mode=1.0; default 0.5) | 0.5 |
| DL_CD | max_decode_steps (CD window) | 64 |
| DL_HBM | hbm_utilization_vllm | 0.6 |

---

## 2. Every run this campaign — config + result + verdict

Baseline for comparison: old champion g4r7b (tp1×dp64) global 125.8, rollout 96.7.
All below on OWNED quota, corrected generation (bucket fix), unless noted.

| Run | Config (delta vs baseline) | Result (global med / mean) | Verdict |
|---|---|---|---|
| g4r9r | baseline: tp1 + bucket fix only | ~127 / stable | corrected-gen reference; bucket fix perf-NEUTRAL + correct → ADOPT |
| g4r9p1 | full host bundle (prefix-off+rr+…) | rollout ~209 | REJECT (net +110s) |
| g4r9s | +prefix_caching=false only | rollout med 77.6 / mean 120.7 | REJECT (median win, MEAN regression: KV spikes) |
| g4r9t | +rr-routing only | rollout ~214-219 | REJECT (2x; dp64 load imbalance) |
| g4r9x | +keepkv+64proc+cd_noeos | 126.7 / 126.9 | NEUTRAL but stable (spike-free) |
| g4r10d | +tp=2 (dp32) | rollout 93.6 / global 115.1 (6-step) | mesh win (short probe) |
| g4r10f | +tp=4 (dp16) | rollout 280 (1 warm step) | REJECT (128 seq/eng + 4-chip all-reduce collapse) |
| g4r11 | tp=2 + padfix, 20-step CONFIRM | **rollout 95.2 / GLOBAL 125.4 / mean 162** | tp=2 ≈ tied w/ champion on median; TAILS dominate mean |
| g4r13 | tp=2 + fp8 KV + verify-timeout (8-step) | RUNNING | testing tail fixes |

DECLINED/NULL (don't rerun): async_scheduling (VOID, sampling-stream change); window-128 (neutral);
tp=4-dp16 (collapse); prefix-off; rr-routing. tp2×dp64 (est. ~127, needs 128 devices, no gain — dp=64
keeps head fan-out). fp8 was excluded then re-requested (g4r13).

### Key numbers to know
- Mesh sweep: tp1×dp64=127 → **tp2×dp32=~125 (best, ~tied)** → tp4×dp16=~328 (collapse). Optimum dp=32.
- The tp-win mechanism: HOST-bound → fewer engines (dp) = less head fan-out. dp is the real knob; tp is
  how you reach it on 64 devices. dp=32 = 64 seqs/engine = sweet spot.
- Device vs host per window (measured): tp1 host 780ms/device 15ms; tp2 host 993ms/device 32ms;
  tp4(g4r4) host 1014ms/device 35ms. Collectives ~1.4% of wall everywhere. HOST is 97-98%.

---

## 3. Patch roster (all self-verifying, fail-open, in `patches/`)

Applied at launch by the launcher's `for P in ...` loop (probes both `patch_X` and `patch_X.py`).
Champion 16 + qwen3-transfer/new: firsttok, engine_kwargs, host_extract (always-on, neutral),
fast_detok (always-on, neutral), dp_rr_routing (gated, REJECTED), sync_keepkv (gated), cd_noeos (gated),
flash_sa (gated, DEFERRED — shim needs local_* splash fields, done), jit_unroll (gated, untested),
patch_sharded_sampling (+ SS_PADFIX padfix), **patch_verify_timeout (reward-tail fix, DL_VTMO)**.

NEW this session (write-ups in each file's docstring):
- `patch_verify_timeout.py` — per-item math_verify timeout (the reward-tail fix). Root cause: pool had
  parsing_timeout/timeout_seconds=None → unbounded sympy. Sets them to VERIFY_ITEM_TIMEOUT=5.
- `patch_sharded_sampling.py` — added SS_PADFIX guard (ignore top_k<=0 padded rows → sharded fast path
  active at tp>1; was silently full-vocab fallback).
- `patch_jit_unroll.py` — jitted weight-sync unroll (tianyu #7, ~-4s sync, UNTESTED, DL_JITU=1).
- `patch_flash_sa.py` — RLConfig splash-block shim (tianyu #8, trainer flash, DL_ATTN=flash, UNTESTED).
- `patch_firsttok.py` — first-token corruption census (echo vs reasoning openings).

---

## 4. THE CORRUPTION FIX (gates everything — always on)

Prefill token-padding buckets were powers of 2 to 1,048,576; real budget 540,672 rounded UP to the
1,048,576 bucket = 94% padding, which CORRUPTS each sequence's first generated token (echo 72% openings).
`VLLM_TPU_BUCKET_PADDING_GAP=32768` (DL_BUCKET_GAP) → max bucket 557,056 (3% pad) → echo 39%. Perf-neutral.
Validity readout: `patch_firsttok` census (echo/reasoning/other) + LENDIST band (65%→83% trunc, osl 2800→3500).
LENDIST unchanged across the fix (content changed, not length) — so the perf ladder stays comparable.

---

### Probe results (2026-08-31, post-handoff-write)
- **fp8 KV: REJECT for RL stack.** vLLM config accepts it but our flax_nnx/RPA path doesn't implement
  fp8 quantize/dequant -> SILENT failure: hollow 22s steps, no completions/LENDIST/FIRSTTOK (the detection
  fingerprint). aysu's serving fp8 works because it's the NATIVE vllm-tpu path (no MaxText/RL weight-sync
  constraint). Bringing fp8 to RL = implement it in RPA + MaxText KV (upstream).
- **TPU_MULTIPROCESS_DP=1: HARD-GATED.** Explicit validator: "not supported with attention DP or on
  Pathways." The built-in multiproc-DP (one engine-core process per DP rank = the Gap-A mechanism) exists
  but requires upstream Pathways compatibility OR rollout off-Pathways. This is the precisely-scoped ask
  for the head-parallelization prize.
- USE_BATCHED_RPA_KERNEL: untested (DL_BRPA knob wired); aysu uses =1 in native serving.
- MODEL_IMPL_TYPE=auto resolves to flax_nnx (same as Meta pins). Correct as-is.
- Slice = 128 devices (2 cores/chip x 64 chips); sampler_fraction=0.5 -> our sweep was on 64 devices.
  DL_SFRAC knob added; g4r12 = tp4 + full-slice (TFRAC=SFRAC=1.0) -> dp32, queued.

## TIANYU GROUND TRUTH UPDATE (2026-08-31 evening)
- New runs rl-156..159 (all Aug 31, all QWEN3 -- no gemma4 from tianyu). rl-159 warm: global 57.5-59.0,
  rollout ~51, **compute_rewards 0.5s** (ours: 8-10s median + spikes to 252s, SAME dataset).
- Their stack now carries OUR phasetime instrumentation (phasetime/step_*.json + PHASETIME lines).
- **LEAD -> patch_fast_reward:** their 0.5s implies cheap numeric/exact-match grading, not per-item sympy.
  Proposal: numeric-first grading (fast path for plain-numeric answers, sympy fallback only for the rest).
  Kills reward median AND spikes at the source. CAVEAT: semantics shift (exact vs symbolic equivalence) --
  spot-check agreement on ~1k samples before adopting.

## SWEEP RESULTS (2026-08-31 evening)
- **g4r16a jit_unroll: WIN, ADOPT.** All 8 steps, exit 0. Clean warm steps 113.5/114.2/117.6/123.8
  (median 123.8) vs same base without = 117-128.6 (median 128.6). ~-4s from sync dispatch collapse.
  113.5 = best clean step ever. Tails unchanged (2 KV spikes, 1 reward spike 128.8). ADD DL_JITU=1 TO BASE.
- **g4r12r2 tp4xdp32 full-slice: FAILED post-step-0, 3 attempts.** Step 0 completes (memory fix works for
  the train program) but dies at the step-1 boundary with error 0101/pjrt.OOM -- hypothesis: weight-sync
  reshard transients on shared devices tip HBM over. Retry staged as g4r12r3 (HBM=0.2 TMBS=2).
- ADC died again ~10:30Z (invalid_rapt), restored 22:30Z via bucket handoff. g4r12r2/g4r16a jobsets were
  uninstalled during the blind window (other agent, presumably).

## QUEUED RUNS — detailed specs for the next agent

All are 8-step probes on OWNED quota. **Base = the optimized tp2×dp32 stack** unless noted:
`DL_TP=2 DL_TMBS=8 DL_PACK=8192 DL_STEPS=8 DL_BUCKET_GAP=32768 DL_CKPT=False DL_KEEPKV=1
DL_MVPROCS=64 DL_NOEOS=1 DL_PADFIX=1 DL_VTMO=5` (env file `gemma4/g4r14b-envs.yaml`).
Launch each with the §1 template (`helm install <run> ... -f gemma4/<run>-envs.yaml`), sequential
(one at a time — good-tenant on the shared owned lane). Fast-fail: drop if not beating current best
(tp2×dp32 optimized ~global median 125 / and check MEAN for tails). Judge on median AND mean.

| Run | Env file | Delta vs base | Tests / hypothesis | Success = |
|---|---|---|---|---|
| **g4r12** | g4r12-envs.yaml | DL_TP=4 DL_DP=-1 DL_TFRAC=1.0 DL_SFRAC=1.0 | tp4×dp32 on FULL 128-dev slice (= g4r4/tianyu mesh). Same 32 rollout engines as base, but trainer on 128 vs 64 devices → maybe ~2x faster trainer. | global < base; esp. trainer < ~17s |
| **g4r16a** | g4r16a-envs.yaml | +DL_JITU=1 | jit_unroll: one jitted slice-all in weight sync vs ~290 per-layer proxy dispatches (tianyu #7, ~-4s sync). Workload-independent → should transfer. | trainer+sync median drops ~2-4s |
| **g4r16b** | g4r16b-envs.yaml | +DL_ATTN=flash DL_SAQ=1024 DL_SAKV=2048 | flash/splash trainer attention vs dense dot-product on 8192-tok packed rows (tianyu #8). Our trainer 17.7s (2x tianyu's). Shim fixed (local_* fields). | trainer median drops ~2-3s; WATCH init crash (splash asserts) |
| **g4r16e** | g4r16e-envs.yaml | +DL_CD=256 | CD window 64→256: 4x fewer head round-trips/rollout. Was neutral at 128 pre-host-understanding; retest. | rollout median drops |

DROPPED/moot: g4r10a/b/c/e (old tp=1 6-step base, superseded by g4r16*). g4r10c (mesh-fraction at tp2 →
dp64 = more engines = worse; the fraction question is answered by g4r12 at tp4→dp32 instead).

DO NOT RERUN (verdicts final): fp8 KV (silent flax_nnx non-support), TPU_MULTIPROCESS_DP=1 (Pathways
hard-gated), async_scheduling (VOID), prefix-off (KV spikes), rr-routing (2x), tp4×dp16 (collapse),
window-128 (neutral). Details in each patch docstring + the report.

THE STRATEGIC ITEM (not an env probe): head parallelization = the ~2-3x lever. Precisely scoped now:
either make TPU_MULTIPROCESS_DP Pathways-compatible upstream, OR process-pool
_process_continue_decode_outputs (tpu_runner.py; plain-array picklable), OR run single-turn under
tunix AgenticRLLearner's ThreadPoolExecutor(max_concurrency) (verify it accepts a 1-turn/no-tool task).

## 5. Open queue / next experiments (priority order)

1. **g4r13 (RUNNING)** — tp=2 + fp8 KV + verify-timeout. If verify-timeout kills the reward tail (mean→median)
   and fp8 kills the rollout-spike tail, this is the real win (mean 162→~125). WATCH: fp8 may be unsupported
   on TPU backend (fast-fail); LENDIST band = validity gate for fp8's numeric change.
2. **jit_unroll (DL_JITU=1)** — untested, ~-4s sync (tianyu #7); workload-independent, should transfer.
3. **flash trainer attn (DL_ATTN=flash)** — untested, targets trainer 17.7s (2x tianyu's); shim fixed.
4. **mesh fraction (DL_TFRAC=1.0)** — Meta main mode; we default 0.5/0.5 (OPD split) without OPD — may be
   leaving half the slice idle per phase. Untested.
5. **20-step CONFIRM** of the best stack once tails are tamed → declare champion honestly on median AND mean.
6. **THE BIG ONE (code, not env): head parallelization (Gap A).** Roofline says ~2-3x headroom, all in the
   single-core head. Options: process-pool `_process_continue_decode_outputs`; revisit
   VLLM_ENABLE_V1_MULTIPROCESSING=0 under Pathways; OR run single-turn under tunix AgenticRLLearner's
   ThreadPoolExecutor(max_concurrency) — that thread pool IS the host parallelism we need (verify it accepts
   a 1-turn/no-tool task first). After the head parallelizes, tp=1 becomes optimal again (fewer collectives).

METHOD (adopted from tianyu): byte-identical step-0 prompts as gate (LENDIST + firsttok), one knob per run,
6-8 step fast-fail probe, judge on median AND MEAN (tails matter for real training), 20-step confirm the winner.

---

## 6. Logging + profiles

- **PHASETIME log line** (per step): `PHASETIME step=N compute_rewards=.. engine_generate=.. generate_and_advantage=.. run_global_step=..`
  rollout=engine_generate, rewards=generate_and_advantage-engine_generate, trainer=run_global_step-generate_and_advantage.
- **LENDIST** (per step): truncation% + osl_mean (validity band). **FIRSTTOK**: echo/reasoning/other census.
- Head log: `kubectl --context=$CTX logs <run>-pathways-head-0-0-XXXX -n $NS -c jax-tpu`.
- **Profiles** (xprof/perfetto): `gs://ubench-logs/alisachen-tpu7x/runs/<run>/phased_profile/decode_only/dp_rank_0/plugins/profile/<ts>/`
  - host trace `gke-...<node>.trace.json.gz` → ui.perfetto.dev (host spans: llm_engine.step window, device_get wait)
  - device xplane `server-...<node>.0.0.xplane.pb` (the ~130MB one hosts dp_rank_0) → xprof (jit__decode_core_impl, %all-reduce/%all-gather/%collective-permute)
- Perfetto key spans: `llm_engine.py:296 step` (window wall), `device_get`/`block_until_ready` (host blocked on proxy).
- Archived head logs: `gs://alisachen/tpu_rl/ironwood/logs/` (g4r7b, g4r8-async-void, g4r8b).
- Report + this handoff mirrored: `gs://alisachen/tpu_rl/ironwood/GEMMA4_PERF_REPORT.md`.

---

## 7. Operational gotchas

- **ADC dies periodically** (`invalid_rapt`): user runs `gcloud auth application-default login` on workstation,
  uploads to `gs://alisachen/tmp/adc.json`; on VM copy to `~/.config/gcloud/application_default_credentials.json`,
  verify, **delete bucket copy immediately** via ADC token (`CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth
  application-default print-access-token) gcloud storage rm ...` — plain rm uses scope-limited compute SA, 403s).
- **Always `export CLOUDSDK_AUTH_ACCESS_TOKEN=...`** before kubectl/helm (gke-gcloud-auth-plugin breaks on invalid_rapt).
- **Checkpoint bug**: stale checkpoint from a same-run_name preempted attempt crashes orbax restore
  (`Unsupported dtype: key<fry>`). Probes use `DL_CKPT=False` + fresh run name.
- **Monitors**: Monitor tool with a poll loop that refreshes the token each cycle and treats empty jobset
  query as "maybe auth" (sleep+continue), not "gone" (an ADC lapse false-alarmed "JOBSET GONE" once).
- **Cluster**: `default` ns owns 0 quota (borrowed, gets reclaimed → ~11 preemptions early on). Owned lane is
  `ubench-regression-tests` (this is why we joined cmcs-ai-ninja-team). Only 4x4x4 + 2x2x1 node pools (no 32-chip).
- **git push**: rebase if rejected (concurrent pushes from other agents/user happen).

---

## 8. tianyu cross-reference (verified)

tianyu-rl-143 launcher configmap (ground truth): qwen3-0.6b, **nvidia/OpenMathInstruct-2 (SAME dataset as us)**,
**SINGLE-TURN** (no agentic — earlier "agentic" claim was WRONG, from vaneja recipe misattribution),
**TP4×DP32**, trainer_devices_fraction=1.0, CD window 128, hbm 0.3, verify procs 64, attention flash.
Newest run tianyu-rl-148: global ~63s, rollout ~56s, rewards 0.3s (their simpler reward/GSM8K-like vs our
heavier OpenMathInstruct math_verify). Their host trace `block_until_ready`=73.5s → HOST-BOUND like us.
Their 12-item ladder = hypotheses, NOT a transferable stack: prefix-off (#1) + rr-routing (#9) REGRESS for us
(workload/mesh differences); the transferable ones are phase-local (jit_unroll #7, flash #8, verify #12).
Meta gemma4 recipe: TP=8/remat=full are 16K-prompt artifacts (not us); mesh-fraction=1.0 + stop-token hygiene
+ RPA re-sweep are the takeaways. All 7 Meta upstream PRs present in our image (3 active, rest present/unused).
