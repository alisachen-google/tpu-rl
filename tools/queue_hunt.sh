#!/bin/bash
# queue_hunt.sh — find a usable Kueue lane with OWNED (non-reclaimable) tpu7x
# quota and (re)launch the remaining benchmark jobs there.
#
#   ./queue_hunt.sh detect          # rank candidate LocalQueues (read-only)
#   ./queue_hunt.sh migrate <ns> <queue>   # move g4r3+g4r4 to that lane
#
# Background (2026-08-25): our default-namespace jobs run on quota BORROWED
# from /tpu-shared-cohort; owners (e.g. ubench-regression-tests) reclaim it and
# Kueue preempts us regardless of priorityClass. A lane is only "safe" if its
# ClusterQueue has nominalQuota for google.com/tpu and we hold RBAC to create
# jobsets+configmaps in its namespace. Charts now template
# {{ .Release.Namespace }}, so `helm -n <ns>` works.
set -u
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
K="kubectl --context=$CTX"

detect() {
  echo "=== LocalQueues cluster-wide ==="
  $K get localqueues -A --no-headers 2>/dev/null | awk '{printf "%-28s %-22s -> %s\n", $1, $2, $3}'
  echo ""
  echo "=== ClusterQueue quota analysis (google.com/tpu) ==="
  for CQ in $($K get clusterqueues --no-headers 2>/dev/null | awk '{print $1}'); do
    $K get clusterqueue "$CQ" -o json 2>/dev/null | python3 -c '
import json, sys
cq = json.load(sys.stdin)
name = cq["metadata"]["name"]
cohort = cq["spec"].get("cohort", "-")
tpu_nom = 0
for rg in cq["spec"].get("resourceGroups", []):
    for fl in rg.get("flavors", []):
        for r in fl.get("resources", []):
            if r["name"] == "google.com/tpu":
                q = str(r.get("nominalQuota", "0"))
                tpu_nom += int(q) if q.isdigit() else 0
used = 0
for fu in cq.get("status", {}).get("flavorsUsage", []):
    for r in fu.get("resources", []):
        if r["name"] == "google.com/tpu":
            t = str(r.get("total", "0"))
            used += int(t) if t.isdigit() else 0
adm = cq.get("status", {}).get("admittedWorkloads", 0)
strat = cq["spec"].get("preemption", {})
rc = strat.get("reclaimWithinCohort", "-"); wc = strat.get("withinClusterQueue", "-")
print(f"{name:24s} cohort={cohort:20s} tpu_nominal={tpu_nom:<5d} tpu_used={used:<5d} admitted={adm:<3} preemption={rc}/{wc}")
'
  done
  echo ""
  echo "=== our RBAC per LocalQueue namespace ==="
  for NS in $($K get localqueues -A --no-headers 2>/dev/null | awk '{print $1}' | sort -u); do
    JS=$($K auth can-i create jobsets.jobset.x-k8s.io -n "$NS" 2>/dev/null)
    CM=$($K auth can-i create configmaps -n "$NS" 2>/dev/null)
    PODS=$($K auth can-i list pods -n "$NS" 2>/dev/null)
    echo "ns=$NS create-jobset=$JS create-configmap=$CM list-pods=$PODS"
  done
  echo ""
  echo "Verdict guide: usable lane = jobset+configmap 'yes' AND its ClusterQueue"
  echo "has tpu_nominal > 0 (owned quota; reclaim cannot evict what you own)."
}

migrate() {
  NS="$1"; Q="$2"
  cd "$(dirname "$0")"
  D=/mnt/disks/scratch/dl-stack/patches-ours
  echo "-- tearing down default-ns copies of remaining jobs"
  helm --kube-context=$CTX uninstall g4r3 g4r4 2>/dev/null
  sleep 20
  PF=""
  for f in patch_vllm_sampler patch_cd_proper patch_lendist patch_phasetime patch_sharded_sampling patch_dump_completions; do
    PF="$PF --set-file workload_patches.${f}=$D/${f}.py"
  done
  PFPACK="$PF --set-file workload_patches.patch_profiler_fix=$D/patch_profiler_fix.py --set-file workload_patches.patch_gc_sync=$D/patch_gc_sync.py --set-file workload_patches.patch_packing_knob=$D/patch_packing_knob.py --set-file workload_patches.patch_packer_fast=$D/patch_packer_fast.py"
  echo "-- g4r4 (combined champion: dp32xtp4 + tmbs=8, perf image) -> $NS/$Q"
  helm --kube-context=$CTX install g4r4 recipe-tpu7x-c/ -n "$NS" \
    -f recipe-tpu7x-c/values.yaml -f recipe-tpu7x/values-4x4x4.yaml -f recipe-tpu7x-c/values-dl.yaml \
    --set queue="$Q" \
    --set workload.image=gcr.io/supercomputer-testing/maxtext-rl-ironwood:head-612fcd2e-gemma4c-perf \
    --set 'workload.envs[4].name=DL_STEPS' --set 'workload.envs[4].value=20' \
    --set 'workload.envs[5].name=DL_TP' --set 'workload.envs[5].value=4' \
    --set 'workload.envs[6].name=DL_RPA' --set 'workload.envs[6].value=1\,8704\,1\,2176' \
    --set 'workload.envs[7].name=DL_TMBS' --set 'workload.envs[7].value=8' \
    $PF \
    --set-file workload_launcher=gemma4_e2b_r1_launcher.sh \
    --set-file workload_config=recipe-tpu7x-c/config.yml
  echo "-- g4r3 (packing eval, packing image) -> $NS/$Q"
  helm --kube-context=$CTX install g4r3 recipe-tpu7x-b/ -n "$NS" \
    -f recipe-tpu7x-b/values.yaml -f recipe-tpu7x/values-4x4x4.yaml -f recipe-tpu7x-b/values-dl.yaml \
    --set queue="$Q" \
    --set workload.image=gcr.io/supercomputer-testing/maxtext-rl-ironwood:2026-08-24-packing \
    --set 'workload.envs[4].name=DL_STEPS' --set 'workload.envs[4].value=20' \
    --set 'workload.envs[5].name=DL_TP' --set 'workload.envs[5].value=8' \
    --set 'workload.envs[6].name=DL_RPA' --set 'workload.envs[6].value=1\,8704\,1\,2176' \
    --set 'workload.envs[7].name=DL_TMBS' --set 'workload.envs[7].value=8' \
    --set 'workload.envs[8].name=DL_PACK' --set 'workload.envs[8].value=8192' \
    $PFPACK \
    --set-file workload_launcher=gemma4_e2b_r1_launcher.sh \
    --set-file workload_config=recipe-tpu7x-b/config.yml
}

case "${1:-detect}" in
  detect) detect ;;
  migrate) migrate "$2" "$3" ;;
  *) echo "usage: $0 detect | migrate <namespace> <localqueue>" ;;
esac
