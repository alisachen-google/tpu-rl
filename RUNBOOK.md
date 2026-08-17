# Runbook: Qwen3-0.6B GRPO RL on TPU v7 Ironwood (bodaborg-tpu7x-nap)

Goal: reproduce the v6e-64 e2e RL flow on tpu7x in `cloud-tpu-shared-capacity`,
**without ubench**, bringing each layer up separately so failures are isolated.

Strategy: ubench's only job was generating the Helm chart in
`recipe-v6e-baseline/` and running `helm install`. We drive the same chart by
hand: recon → chip smoke test → Pathways bring-up → 2-step training smoke →
full run. One new component per phase.

All kubectl/helm commands run from the workstation (`alisachenspecial`), which
already has cluster credentials:

```bash
kubectl config use-context gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
```

---

## Phase 0 — Recon (read-only, 10 min)

Fill the Decision Table below from these commands before touching anything.

```bash
# 0.1 TPU node labels: accelerator type, topology, chips/host
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator \
  -o custom-columns='NAME:.metadata.name,ACCEL:.metadata.labels.cloud\.google\.com/gke-tpu-accelerator,TOPO:.metadata.labels.cloud\.google\.com/gke-tpu-topology,NODEPOOL:.metadata.labels.cloud\.google\.com/gke-nodepool,CHIPS:.status.allocatable.google\.com/tpu'

# 0.2 Required controllers — JobSet CRD must exist; Kueue may or may not
kubectl get crds | grep -E "jobset|kueue|pathways"
kubectl get clusterqueues,localqueues -A 2>/dev/null
kubectl get priorityclasses | grep -vE "system-"

# 0.3 What do working jobs on THIS cluster look like? (best source of truth)
kubectl get jobsets -A
kubectl get jobset <some-jobset> -n <ns> -o yaml > example-jobset.yaml
#   → note: pathways server image tag, accelerator labels, queue name, ports

# 0.4 CPU nodepool capacity for the head pod (chart requests 28 CPU / 130G)
kubectl describe node -l cloud.google.com/gke-nodepool=cpu-np | grep -A5 Allocatable

# 0.5 Which GCS buckets can pods write? Check node SA + an existing job's output path
kubectl get nodes -o jsonpath='{.items[0].metadata.annotations}' | tr ',' '\n' | grep -i service-account
```

### Decision Table (fill in)

| Key | v6e value (baseline) | tpu7x value |
|---|---|---|
| accelerator label | `tpu-v6e-slice` | ← from 0.1 (likely `tpu7x`) |
| topology label | `8x8` | ← from 0.1 (pick smallest available slice first) |
| chips per host | 4 | ← from 0.1 |
| worker pod count | 16 | topology chips ÷ chips-per-host |
| pathways `instanceType` | `tpuv6e:8x8` | `tpu7x:<topo>` — confirm string in example-jobset |
| pathways server image | `.../pathways/server:20260623-jax_0.10.1` | ← copy tag from a working tpu7x jobset (must have Ironwood support) |
| workload image | `maxtext_post_training_nightly:2026-07-15` | latest nightly (needs libtpu with tpu7x support) |
| Kueue queue | `multislice-queue` | ← from 0.2/0.3, or drop the label if no Kueue |
| priority class | `very-high` | ← from 0.2, or drop |
| output bucket | `gs://ubench-logs` | writable bucket, e.g. `gs://alisachen/tpu_rl/runs` (verify cluster SA has access!) |
| checkpoint `load_parameters_path` | `gs://mohitkhatwani_multipods/qwen3-0.6b/...` | verify readable from this project; else copy into your bucket |

---

## Phase 1 — Raw chip smoke test (no Pathways, no Helm)

Validates: scheduling onto tpu7x nodes, TPU driver, libtpu-Ironwood compat.
Single pod, direct JAX (bypasses the entire Pathways stack):

```yaml
# 1-tpu-smoke.yaml  (fill selectors from Decision Table)
apiVersion: v1
kind: Pod
metadata: {name: tpu7x-smoke}
spec:
  restartPolicy: Never
  nodeSelector:
    cloud.google.com/gke-tpu-accelerator: <ACCEL_LABEL>
    cloud.google.com/gke-tpu-topology: <TOPO>
  containers:
  - name: jax
    image: <WORKLOAD_IMAGE>          # maxtext nightly includes libtpu
    command: ["python3","-c","import jax; print(jax.devices()); print('chip count:', jax.device_count())"]
    env: [{name: JAX_PLATFORMS, value: "tpu,cpu"}]
    resources: {limits: {google.com/tpu: <CHIPS_PER_HOST>}}
```

Pass = it prints TpuDevice entries. Fails here → node/driver/image problem,
nothing to do with Pathways or MaxText. (Pod is single-host; that's fine —
this only validates chip access.) `kubectl delete pod tpu7x-smoke` after.

## Phase 2 — Pathways runtime bring-up (Helm, trivial payload)

Validates: JobSet gang-start, RM/proxy/worker registration, IFRT proxy path.
Use the chart but replace the launcher with a trivial payload:

```bash
# 2-devices.sh — minimal launcher
cat > /tmp/devices.sh <<'EOF'
#!/bin/bash
python3 -c "import pathwaysutils, jax; print(jax.devices()); print('devices:', jax.device_count())"
EOF

cd recipe-tpu7x/   # your edited copy of the chart
helm template rl-smoke . -f values.yaml --set-file workload_launcher=/tmp/devices.sh \
  --set-file workload_config=qwen3-0.6b.yaml > rendered.yaml   # inspect first!
kubectl apply -f rendered.yaml
```

Debug order when it fails:
1. Pods `Pending` → `kubectl describe pod` (Kueue admission? nodeSelector
   mismatch? NAP still provisioning nodes — the `-nap` cluster autoprovisions,
   first scheduling can take several minutes)
2. Workers CrashLoop → `kubectl logs <head-pod> -c pathways-rm` (do workers
   register? instance_type string wrong?)
3. Head hangs → `kubectl logs <head-pod> -c pathways-proxy`, check
   `JAX_BACKEND_TARGET` DNS name resolves (JobSet `enableDNSHostnames`)

Pass = head prints the full expected device count (all workers' chips).

## Phase 3 — Training smoke (2 steps)

Validates: checkpoint restore, vLLM engine init on Ironwood, one full GRPO step.
Use the real `launcher.sh` with overrides appended to the python command:

```
steps=2 num_batches=2 enable_checkpointing=false
```

Expectations / things to watch (from the v6e baseline run):
- Checkpoint restore ~6 s ×2. Fails → bucket permissions from this project.
- vLLM "Pathways dummy weight loading" took **34 min** on v6e. First tpu7x run
  will also pay full compilation — point `JAX_COMPILATION_CACHE_DIR` and
  pathways `gcsLocation` at your own bucket so later runs are warm.
- First `sampler-trainer: ... pearson=` line should be >0.8. Much lower on
  tpu7x than v6e → numerics/kernel issue worth flagging.
- `Train loop finished in: ~1s` lines appearing = the trainer half works.

## Phase 4 — Full run (steps=30) + manual artifact capture

ubench captured artifacts automatically; manually it's:

```bash
RUN=rl-tpu7x-$(date +%m%d-%H%M)
helm install $RUN . -f values.yaml --set-file workload_launcher=launcher.sh \
  --set-file workload_config=qwen3-0.6b.yaml
kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=$RUN -c jax-tpu --max-log-requests=1 \
  | tee /tmp/$RUN-head.log
gcloud storage cp /tmp/$RUN-head.log gs://<OUTPUT_BUCKET>/$RUN/logs/
# checkpoints + tensorboard go to base_output_directory automatically
helm uninstall $RUN   # frees the chips
```

---

## Chart edits: v6e baseline → tpu7x (make `recipe-tpu7x/` from the baseline)

`values.yaml`:
1. `targetNodeSelector.cloud.google.com/gke-tpu-accelerator` → tpu7x label
2. `targetTopology`, `nodes`, `chips`, `workload.tpus/gpus` → per Decision Table
3. `pathways.instanceType` → `tpu7x:<topo>`
4. `pathways.serverImage` / `proxyServerImage` → Ironwood-capable tags
5. `workload.image` → current nightly
6. **Delete all `proxyExtraArgs` XLA flags initially** — they are v6e
   sparse-core tuning; start Ironwood with defaults, re-tune later
7. `queue` / `kueue_priority_class` → per Decision Table (or remove)
8. `gcsLocation`, `JAX_COMPILATION_CACHE_DIR`, `workload.envs[ARTIFACT_DIR]`
   → your writable bucket
9. `cluster_name: bodaborg-tpu7x-nap`, `cloud_region: us-central1`

`launcher.sh`: only the override string (steps, checkpointing) per phase.

### Config deltas vs the v6e baseline recipe (current team config)

| Knob | v6e baseline we extracted | Current config | Meaning at num_generations=4 |
|---|---|---|---|
| `batch_size` | 32 | **256** (down from a previous 480) | 256 prompts → **1024 sequences/step** (was 128) |
| `train_micro_batch_size` | 4 | **32** | 1024 ÷ 32 = 32 accumulation micro-batches per update |
| `rollout_micro_batch_size` | 32 | **256** | all prompts of a step handed to vLLM in one chunk |

Implications:
- **Step time scales ~8×** on the rollout side vs the v6e-32 baseline unless
  sampler throughput rises to match. With 1024 sequences and `max_num_seqs=64`,
  vLLM works through ~16 waves per step — on Ironwood (192G HBM/chip) consider
  raising `max_num_seqs` / `hbm_utilization_vllm` to use the extra headroom.
- **Baseline curves from the v6e run are no longer apples-to-apples** for
  absolute step time; still valid for: reward *trend*, pearson (~0.87),
  completion lengths.
- Divisibility: keep `batch_size` divisible by `train_micro_batch_size` and by
  the data-parallel mesh dim; 256/32 is clean for 32-, 64-chip splits.
- Smoke tests T5/T6 must use THESE values (shapes determine what gets
  compiled/cached): micro-batch of 32 sequences is the trainer's compiled
  shape, 256-prompt rollout chunk is the sampler's workload shape.

Head pod: if cpu-np nodes are smaller than 28 CPU/130G, lower the requests in
`templates/pathways-jobset.yaml` (0.6B model needs far less than that).

## Known failure modes to expect

| Symptom | Likely cause |
|---|---|
| Workers Pending forever | NAP hasn't provisioned; or topology label value has no matching nodepool shape |
| Kueue "not admitted" | wrong/missing queue name, quota exhausted in shared cluster |
| RM rejects workers | `instance_type` string wrong for tpu7x |
| Proxy up, jax.devices() hangs | DNS hostname mismatch (release name changed → env var strings in jobset must match) |
| vLLM init crashes | nightly image too old for Ironwood (libtpu), or tpu_inference lacks tpu7x support — try newer nightly |
| GCS 403s from pods | cluster node SA lacks access to your bucket — grant `roles/storage.objectAdmin` on `gs://<bucket>` to the node SA |

## Fix Catalog — Ironwood bring-up (2026-08-03, all applied)

Every fix discovered during the smoke ladder + T6 bring-up. The chart/launcher
in recipe-tpu7x/ and image 2026-08-03b already contain all of them.

| # | Symptom | Root cause | Fix (where) |
|---|---------|-----------|-------------|
| 1 | `Backend 'proxy' is not in known backends` | pathwaysutils ≥0.1.11 needs explicit init | `pathwaysutils.initialize()` (smoke scripts; train_rl does it itself) |
| 2 | JobSet rejected: `restartStrategy` unknown | older JobSet CRD on this cluster | `failurePolicy.maxRestarts: 0` (template) |
| 3 | `no PriorityClass "high"` (worked hours earlier) | priority classes churn in this cluster | pin `medium` (values) |
| 4 | pydantic assert `VLLM_ENABLE_V1_MULTIPROCESSING` | required =0 under Pathways | export in launcher |
| 5 | `Unknown backend cpu. Available: ['proxy']` | newer tpu_inference stages weights on cpu backend | `JAX_PLATFORMS="proxy,cpu"` (launcher) |
| 6 | RM dies: "Quick-restart requested but unsupported" | stale Pathways scratch state reused across runs | per-render UUID suffix on `gcs_scratch_location` (template) |
| 7 | worker dies with same quick-restart error | hostPath /tmp leaks worker state across pods on reused nodes | `emptyDir` for shared-tmp (template) |
| 8 | silent exit-1, no error logged | `set -e` kills launcher at `wait` before epilogue | `EXIT_CODE=0; wait $PID \|\| EXIT_CODE=$?` + DEBUG-HOLD (launcher) |
| 9 | `CONFIG ERROR: 'dataset_type' not in ...` | maxtext main's typed RLConfig dropped v6e-era keys | removed `dataset_type`, `steps` from overrides (num_batches governs) |
| 10 | `RLConfig has no attribute use_mrope/mrope_section` | qwen3.py model fields missing from RLConfig schema | added `MultimodalGeneral` mixin to RLConfig (types.py patch, image ≥03b) — UPSTREAM BUG, report to maxtext |
| 11 | vLLM engine inits with 0 workers (silent degenerate) | Pathways exposes declared/virtual devices before workers exist | device-count gate in smoke; execution queues until workers register |
| 12 | new head crashes when redeployed quickly | hostNetwork port collision with draining prior pods | wait until NO prior pathways pods exist before apply |
| 13 | NAP won't create multi-host tpu7x pools | autoprovisioning covers single-host (2x2x1) only here | multi-host slices must be admin-provisioned (reservation cloudtpu-20260710003900-159478293, READY, us-central1-c) |
| 14 | tensorboardX 412 at exit when reusing run_name | GCS generation mismatch on reused events file | use a fresh run/release name per attempt |
| 15 | head pod Pending: `node(s) had untolerated taint` on cpu-np | cpu-np nodepool gained a NoSchedule taint (2026-08-14) | toleration `cloud.google.com/gke-nodepool op=Exists` on head pod (template) |
| 16 | gemma ckpt restore: zarr NOT_FOUND under Pathways | original gemma4-e2b ckpt is OCDBT; Pathways-compat restore needs raw TensorStore | converted to raw at `gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw/0/items` (541 arrays) |
| 17 | jobset Inadmissible: `LocalQueue multislice-queue doesn't exist` | LocalQueues churn like PriorityClasses; queue removed from `default` ns (2026-08-17) | `queue: default` in values.yaml; check `kubectl get localqueues` before each campaign |
| 18 | `ValueError: prompt_logprobs is not supported for multimodal models` (gemma4) | gemma4 HF config registers multimodal; tpu_runner blocks GRPO's prompt logprobs | team route: `vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}'` + `vllm_additional_config.maxtext_config` (per run_gemma4_e4b_rl.sh in image); sampler then does DIRECT MaxText→MaxText weight sync (no vLLM weight mapping needed) |
| 19 | every rollout `MAX_CONTEXT_LIMIT_REACHED`, 4096 tokens, no EOS | agentic rollout stringifies raw message dicts (numpy reprs) into the prompt — gemma4-it rambles to budget; UPSTREAM BUG (decoded prompt showed `{'role':..., 'content': array(...)}` as literal text) | drop `rl.use_agentic_rollout`; non-agentic path builds prompts via process_data + chat template correctly. Also add `data_template_path` + `chat_template_path` (gemma-3-27b) per team recipe |
| 20 | `'hf_name' not in ...` config error | e4b reference script targets different RLConfig schema (drift, cf. Fix #9) | drop `hf_name=default` from overrides |
| 21 | trainer OOM step 1: `HLO temporaries (130.13G) exceeds available HBM (94.74G)` | gemma 262k-vocab logits: tmbs 4 × 8192 × 262144 fp32 ≈ 34G ×~4 temporaries, NOT batch-sharded across trainer devices; Ironwood exposes ~95G usable per core-device | `train_micro_batch_size=1` (matches team e4b recipe and the GB200 pd=2 freeze); tmbs=2 may fit (~65G) — P1 scout question |
| 22 | `FAILED_PRECONDITION: Cache key mismatch between PW client and server while compiling` ~3 min in (t6gn4/5, t6g11/12, 2026-08-17 21:40–22:07) | ANOTHER USER'S pathways head (vaneja's, crashlooping) co-resident on the single cpu-np node 21:15–22:20; hostNetwork + fixed ports 29000/29001 → our client dialing 127.0.0.1:29000 reached THEIR proxy (different Pathways build) → client/server version skew. Multi-tenant generalization of Fix #12. Ruled out first: image digests (matched registry, tags from 07-30), client XLA flags (log diff clean), cgroup limits (t6g12 failed with originals) | Before every launch: `kubectl get pods -A -o wide \| grep pathways \| grep <cpu-node>` must show NO foreign pathways head. Long-term ask: per-run ports or drop hostNetwork in the chart |

## Milestones achieved (2026-08-03)
- T0-T4 PASS; T5 partial (off-prod paths retired); **T6-mini PASS**: full
  train_rl GRPO loop on tpu7x 2x2x1 — 17 min e2e, pearson 0.997 (vs 0.87 on
  v6e!), first train loop 140 s (compile) then 2-7 s, checkpoint saved, 0 restarts.
- 30-step e2e launched at 2x2x1 scale (batch 8). Full-scale (4x8x8, batch 256)
  pending slice re-provisioning by team.
