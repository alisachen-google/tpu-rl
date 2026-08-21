#!/bin/bash
# Gemma4 optimization sweep (2026-08-21) — single-variable deltas vs g4r1c
# (271.2 s median: DL_TP=8, DL_RPA=1,8704,1,2176, CD=64, EOS int=1, tmbs=1).
# Each scout: 8 steps -> warm n=5 for a steps>=3 median. Compile cache is warm
# from g4r1c, so cold step ~6-8 min instead of 27.
#
# Phase 1 (parallel, chart-b + chart-c):
#   g4s1  CD window 128 + EOS-check interval 8   (rollout dispatch lever)
#   g4s2  tmbs=2                                  (trainer lever; pretrain
#         recipe fits per-device-batch 2 @ 8192 -> expected safe)
# Phase 2 (after phase 1 reads, same charts):
#   g4s3  RPA tile: kernel default (no env)       (was 1,8704,1,2176 worth anything?)
#   g4s4  combine winners
set -e
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
cd ~/tpu_rl/ironwood
D=/mnt/disks/scratch/dl-stack/patches-ours
PF=""
for f in patch_vllm_sampler patch_cd_proper patch_lendist patch_phasetime patch_sharded_sampling patch_dump_completions; do
  PF="$PF --set-file workload_patches.${f}=$D/${f}.py"
done
IMG=gcr.io/supercomputer-testing/maxtext-rl-ironwood:head-612fcd2e-gemma4c-perf

launch() { # $1=release $2=chart $3.. extra --set args
  local REL=$1 CHART=$2; shift 2
  helm --kube-context=$CTX install $REL $CHART/ \
    -f $CHART/values.yaml -f recipe-tpu7x/values-4x4x4.yaml -f $CHART/values-dl.yaml \
    --set workload.image=$IMG \
    --set 'workload.envs[4].name=DL_STEPS' --set 'workload.envs[4].value=8' \
    --set 'workload.envs[5].name=DL_TP'    --set 'workload.envs[5].value=8' \
    --set 'workload.envs[6].name=DL_RPA'   --set 'workload.envs[6].value=1\,8704\,1\,2176' \
    "$@" $PF \
    --set-file workload_launcher=gemma4_e2b_r1_launcher.sh \
    --set-file workload_config=$CHART/config.yml
}

case "${1:-phase1}" in
phase1)
  launch g4s1 recipe-tpu7x-b \
    --set 'workload.envs[7].name=DL_CD'      --set 'workload.envs[7].value=128' \
    --set 'workload.envs[8].name=DL_EOS_INT' --set 'workload.envs[8].value=8'
  launch g4s2 recipe-tpu7x-c \
    --set 'workload.envs[7].name=DL_TMBS'    --set 'workload.envs[7].value=2'
  ;;
phase2)
  # g4s3: no DL_RPA (kernel default tile) — overwrite env slot with a no-op
  helm --kube-context=$CTX uninstall g4s1 g4s2 2>/dev/null || true
  sleep 30
  launch g4s3 recipe-tpu7x-b \
    --set 'workload.envs[6].value=' \
    --set 'workload.envs[7].name=DL_CD'      --set 'workload.envs[7].value=128' \
    --set 'workload.envs[8].name=DL_EOS_INT' --set 'workload.envs[8].value=8'
  ;;
esac
