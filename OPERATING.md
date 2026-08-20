# Operating guide — gemma4-e2b RL on TPU v7x (bodaborg-tpu7x-nap)

How to launch, watch, and read a run with this repo. Companions: `RUNBOOK.md` (Fix Catalog
#1–22, bring-up history), `gemma4/G4R1_RUN_PLAN.md` (the current campaign + log),
`gemma4/GEMMA4_PERF_REPORT.md` (results), `tuning/` (rollout-tuning and manifest analyses).

---

## 0. Access (VM `ac-ml-perf`)

**Approved path (prior campaign, re-validated 2026-08-20):** your `alisachen@google.com` ADC, minted on
the **cloudtop `alisachenspecial`** (NOT the MacBook — that copy is CAA-blocked) and copied here; `gke-gcloud-auth-plugin` mints cluster tokens from it on this VM.
```bash
# cloudtop alisachenspecial (browser login), then copy:
gcloud auth application-default login
gcloud compute scp ~/.config/gcloud/application_default_credentials.json \
  ac-ml-perf:~/.config/gcloud/application_default_credentials.json --zone us-central1-b --project gke-aishared-gsc-dev
# VM: verify
gke-gcloud-auth-plugin --use_application_default_credentials | head -c 80   # JSON => OK
```
When access breaks with `invalid_rapt` / "Reauthentication failed" → the RAPT expired (~daily): redo the
two cloudtop lines. That is NOT the Context-Aware-Access block (`"Account restricted"` / "Access was
blocked by Context Aware Access") — that one only hits `gcloud auth login`-style user creds refreshed
here; ADC copied from alisachenspecial has been accepted on this VM (the MacBook's ADC is the one CAA rejects).

| Piece | Value |
|---|---|
| kubectl | `~/bin/kb` = kubectl pinned to bodaborg. Tries `bodaborg-adc` (gke-gcloud-auth-plugin + ADC), then `bodaborg-adc2` (`~/bin/adc-exec-cred.sh`), then the SA-token context. **Always use it** — a background process on this VM rewrites `~/.kube/config` and flips the current context to a GPU cluster; never `use-context`. |
| fallback identity | K8s ServiceAccount `default/alisa-runner` via `~/.k8s-token` (tokenFile). Namespaced Role: pods, pods/log, configmaps, services, jobs, jobsets. No secrets (→ `helm template \| kubectl apply`), no nodes/queues. Renewal needs `container.serviceAccounts.createToken` (removed by the 2026-08-19 RBAC rollout) — or the vendor pod `tools/token-vendor-pod.yaml`, which publishes the SA's projected token to `gs://alisachen/tpu_rl/auth/` for `tools/refresh_token.sh`. |
| GCS | works via the VM service account (read; runs/ubench-logs artifacts readable, `gs://alisachen` readable). |
| GitHub | PAT in `~/.git-token`; commits authored as `alisachen-google` only. |

---|---|
| kubectl | `~/bin/kb` = `kubectl --context=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap`. **Always use it** — a background process on this VM rewrites `~/.kube/config` and flips the current context to a GPU cluster. |
| identity | K8s ServiceAccount `default/alisa-runner`; bearer token in `~/.k8s-token` (`~/.kube/config` uses `tokenFile`). Namespaced Role: pods, pods/log, configmaps, services, jobs, jobsets — full CRUD. **No secrets** (hence `helm template \| kubectl apply`, never `helm install`), no nodes/queues/priorityclasses (read those from a machine with user creds). |
| token renewal | on a machine with cluster creds: `kubectl create token alisa-runner -n default --duration=168h > /tmp/tok` then `gcloud compute scp /tmp/tok ac-ml-perf:~/.k8s-token --zone us-central1-b --project gke-aishared-gsc-dev`. The API server caps the lifetime (observed 48 h). |
| gcloud user auth | **impossible on this VM** — Context-Aware Access blocks token refresh for @google.com from a non-corp device. GCS works via the VM service account. |
| GitHub | PAT in `~/.git-token`; credential helper configured; commits authored as `alisachen-google` only. |

---

## 1. Launch

```bash
cd ~/tpu-rl
tools/launch_run.sh r1n-4x4x4        # prints the run name, e.g. g4r1n-08200548
tools/watch_run.sh                   # follow it (admission → pods → steps)
```

| Profile | What it runs |
|---|---|
| `r1n-4x4x4` | **control benchmark**: native `Gemma4ForCausalLM` sampler + full tianyu stack, 4x4x4 (64 chips), 30 steps, batch 256×8, 8192/4096 tokens, tmbs=1, tp=4/dp=16 |
| `r1np-4x4x4` | control + maxtext PR #4409 chunked actor log-probs, **tmbs=2 / chunk 512** |
| `r1-smoke` | same code at 2x2x1, 2 steps, batch 8 — run this after any code/patch change |
| `r1-smoke-pr` | PR #4409 rung at 2x2x1, tmbs=2 / chunk 512 |

Extra `--set` args pass through, e.g. `tools/launch_run.sh r1n-4x4x4 --set kueue_priority_class=low`.
The script pins what currently works on this cluster: LocalQueue `multislice-queue`, PriorityClass
`medium` (`high`/`very-high` were deleted — Fix #3 churn), and Pathways ports shifted to 291xx so a
foreign head co-resident on the cpu-np node cannot collide (Fix #12/#22).

**Pre-flight** (30 s, run before a 4x4x4 submit):
```bash
~/bin/kb get pods -o wide | grep -E "pathways|worker" | grep -v <your-run>   # foreign heads on cpu-np?
~/bin/kb get jobsets | grep -v true                                          # anything of ours still live?
# needs user creds (ask, or run on a workstation):
kubectl get nodes -l cloud.google.com/gke-tpu-topology=4x4x4                  # is a 16-node pool present?
kubectl get localqueues -A ; kubectl get priorityclasses | grep -v '^system'  # queues/classes still exist?
```

## 2. Watch

`tools/watch_run.sh [run]` streams: Kueue admission/eviction events, suspend/pod/restart state, and
the log lines that matter (patch banners, `LENDIST`, `PHASETIME`, reward pool, tracebacks, kills).

Timeline to expect at 4x4x4 (cold caches): pods Running ~2 min after admission → checkpoint restore
+ vLLM engine init ~10 min → rollout of 2048 completions ~10 min → **train_step compile 20–35 min**
(unrolled 35 layers × 262k vocab × 128-way SPMD; one-time per program shape) → `PHASETIME step=0`,
then warm steps.

## 3. Read the gates

```bash
tools/run_gates.sh <run>        # LENDIST, PHASETIME, rewards/kl/perf scalars from GCS
python3 tools/tb_scalars.py <events-file-or-gs-path> '<tag-regex>'   # raw scalar reader (no TF dep)
```

| Gate | Pass condition |
|---|---|
| Patches applied | every `*PATCH: APPLIED` banner in the head log (7 for control, 9 for the PR rung) |
| Single-pool placement | all 16 workers on one `gke-tpu-*` nodepool (`kb get pods -o wide`) |
| Weight push correct | coherent EOS-terminating completions, `actor/train/kl ≈ 0`, rewards > 0. Gibberish + ~97% truncation = the sampler is running on wrong weights |
| No prompt truncation | LENDIST `isl_mean` ≪ `max_prefill_predict_length` |
| Steps completed | `perf/train/global_step_time` scalars exist; warm median over steps ≥ 2 is the benchmark number |

Artifacts: `gs://ubench-logs/alisachen-tpu7x/runs/<run>/{tensorboard,phasetime,lendist,checkpoints}`;
archived head logs: `gs://alisachen/tpu_rl/ironwood/logs/`.

## 4. Tear down

```bash
~/bin/kb delete jobset <run> --wait=false
until [ -z "$(~/bin/kb get pods --no-headers | grep <run>)" ]; do sleep 5; done
```
Always drain before the next 4x4x4 submit (frees chips and avoids port/DNS collisions).

---

## 5. Known failure modes (2026-08-19/20 campaign)

| Symptom | Cause | Action |
|---|---|---|
| Pods never created: `no PriorityClass with name high was found` | PriorityClasses churn on this cluster; `high`/`very-high` deleted | use `medium` (the launcher default); probe with a server dry-run pod before assuming a class exists |
| Job admitted, then `Preempted … due to reclamation within the cohort` minutes later | our ClusterQueue `default` only **borrows** tpu7x quota; `ubench-regression-tests` reclaims its own | retry (JobSet `maxRestarts: 5` absorbs it), or get quota in a queue that owns it (`priority-dev` ClusterQueue is idle but we lack RBAC there) |
| Head SIGTERMed (exit 143) mid train-compile, no traceback, no Kueue event | node drain — `optimize-utilization` autoscaler scale-down or node repair | pods now carry `cluster-autoscaler.kubernetes.io/safe-to-evict: "false"`; confirm identity in Cloud Logging (`cluster-autoscaler-visibility` decisions, or audit `pods.delete` → `principalEmail`) |
| Worker job `FailedCreate: vpod.kb.io … missing selector: cloud.google.com/gke-nodepool` | the 4x4x4 pool is absent/incomplete, so the topology webhook cannot place followers | wait for the pool to be re-provisioned (admin/NAP); nothing to fix in the manifest |
| Gibberish completions at `rollout_tensor_parallelism > 1` | fused qkv/gate_up must be **interleaved per TP shard** for tpu_inference's merged linears | fixed by `patch_gemma4_tp_fusion` + `GEMMA4_NATIVE_TP` (= `R1_TP`) |
| `RLTrainingConfig … unexpected keyword 'compute_logps_chunk_size'` | image tunix predates chunked logps | `patch_tunix_chunked_logps` (ships the tunix half of PR #4409) |
| `HLO temporaries (113–130G) exceeds available HBM (94.7G)` | 262k-vocab logits at tmbs > 2 | tmbs=1 (control) or tmbs=2 + chunk 512 (PR rung) — Fix #21 |
| `start_trace called while a trace is already being taken` | tunix profiler + `PHASED_PROFILING_DIR` together | run only one profiler (we use phased) |

## 6. What's in the repo

```
chart/                 helm chart (JobSet + ConfigMaps); values-*.yaml overlays per shape/rung
chart/patches/         runtime patches mounted at /workload/patches and applied by the launcher
gemma4/*_launcher.sh   the launcher scripts (one per route/rung); env-driven (R1_STEPS, R1_BATCH,
                       R1_TMBS, R1_TP, R1_LOGPS_CHUNK, R1_RPA, USE_BATCHED_RPA_KERNEL, RL_VLLM_ENGINE_KWARGS)
tools/                 launch_run.sh, watch_run.sh, run_gates.sh, tb_scalars.py
renders/               exact manifest submitted for every run (audit trail)
tuning/                aysu rollout-tuning analysis, tianyu-rl-17 manifest comparison
logs/                  head logs captured from runs of interest
```
