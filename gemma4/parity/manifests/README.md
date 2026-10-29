# End-to-end recipes (apply directly with kubectl)

Each YAML is self-contained: two ConfigMaps (launcher script + patches) and one Kueue-managed JobSet
(1 Pathways head on `cpu-np`, 16 TPU workers = one tpu7x 4x4x4 slice, 64 chips). Nothing else has to be
installed; every patch is applied by the launcher inside the pinned image at pod start.

| manifest | what it runs | outcome on 2026-10-27 |
|---|---|---|
| `g4pt1-seed42.yaml` | GSM8K accuracy-parity run, seed 42: gemma4-E2B-it, GRPO + token TIS, 250 updates, eval every 20 steps on the 1319-question test set (scorer-v2) | step-250 **92.49 %** vs GPU 92.42 % — gate ≥ 91.42 passed |
| `g4pt2-seed43.yaml`, `g4pt3-seed44.yaml` | same recipe, seeds 43 / 44 (for the 3-seed mean) | queued |
| `g4ucq2-optimized-24s.yaml` | the perf-optimized recipe (OpenMathInstruct-2, 256×8 rollouts, 2048-token response cap, round-robin routing, approx-top-k, async reward), 100 updates | 23.9 s/step median; check_numbers 0.512 in the GPU band 0.50–0.57, KL 0 — sustained parity |

## Run

```bash
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
NS=ubench-regression-tests                       # LocalQueue multislice-queue -> ClusterQueue ubench-regression-tests
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json

kubectl --context $CTX -n $NS apply --server-side --force-conflicts -f g4pt1-seed42.yaml
kubectl --context $CTX -n $NS get jobset g4pt1 -w                       # Suspended -> admitted -> running
POD=$(kubectl --context $CTX -n $NS get pods | awk '/g4pt1-pathways-head/{print $1}')
kubectl --context $CTX -n $NS logs -f $POD -c jax-tpu | grep -E "EVAL-V2ACC|EVAL-BREAKDOWN|Restored params|TIS-ACTIVE"
```

Outputs land under `gs://ubench-logs/<jobset-name>/`: `evals/eval_<step>.jsonl` (per-question
scorer-v2 rows), `checkpoints/actor/<step>/` (every 20 steps), tensorboard (g4ucq2 only).

Score a parity run against the GPU reference curve and draw the figures:

```bash
gsutil -m cp "gs://ubench-logs/g4pt1/evals/eval_*.jsonl" evals/
python3 ../tools/make_parity_figures.py --gpu <aggregate_summary.json> --tpu "seed42=evals:<head-pod-log>" --out figs
```

Score the optimized recipe (reward band from the GPU verl reference):

```bash
gsutil -m cp -r "gs://ubench-logs/g4ucq2/tensorboard/*" tb/ && python3 ../tools/tb_parity.py tb/events.out.tfevents.*
```

## Cluster rules that the manifests already encode

* **One 64-chip JobSet at a time** in this namespace (concurrent ones preempt each other).
* **120-minute limit per admission** (cluster `time-limit-controller`): a 250-step parity run needs one
  resume. The run checkpoints every 20 steps and restores on restart (RNG-free checkpointables,
  rollout-engine sync after restore). To resume after a deactivation: `kubectl delete jobset g4pt1`,
  clear any uncommitted step dir (`../tools/clean-uncommitted-ckpts.py g4pt1`), re-apply the same
  manifest. `../tools/chain-seeds3.sh` does all of this unattended for seeds 42 → 43 → 44.
* **Head pod uses host networking on ports 29140/29141** (not the Pathways defaults 29000/29001,
  which other tenants' heads hold on the shared CPU nodes).
* A fresh run name needs an empty `gs://ubench-logs/<name>/checkpoints/`; otherwise it resumes.

## Regenerate a manifest

`../build_g4pt.py` builds `g4pt{1,2,3}.yaml` from the full-recipe source manifest (`g4pf.yaml`, not
in the tree) with `CKPT=1`; `../build_g4ucq2.py` derives the optimized-recipe rerun from `g4ucq.yaml`.
The embedded patches are the files under `../patches/`.

**2026-10-29:** every manifest here now sets `--xla_tpu_dvfs_p_state=6` (7 is not allowed on this cluster).
`g4ucq3-optimized-24s-pstate6.yaml` is the 24 s recipe rerun at p-state 6; `g4pl10-latest-image-10step.yaml`
is the 35 s recipe on the latest image (`2026-09-22-head`, 10 updates) for the perf-regression check;
`g4pl10a/b/d-*` are its variants (legacy sync, re-converted checkpoint, native vLLM model). None of the four
runs on that image: see `../../GEMMA4_E2B_LATEST_IMAGE_STATUS.md` (rollout weights gibberish, missing gemma4
vLLM weight mapping, `local_sa_*` None crash).
