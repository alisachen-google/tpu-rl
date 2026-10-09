#!/usr/bin/env bash
# PR #1193 live check: with the re-scoped branch, --kv-canary log on TPU must fail fast with a clear message (no stall, no segfault).
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; OUT=$R/fix_validate_v2
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [canary-refuse] $*"; }
while pgrep -f 'phase_fix_validate_v[3]' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main && git bundle create /tmp/canary2.bundle tpu-kv-canary-fused-pool 2>/dev/null
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/canary2.bundle sa_101411256518727260534@10.128.15.217:/tmp/canary2.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/canary2.bundle; $GA fetch -q /tmp/canary2.bundle +tpu-kv-canary-fused-pool:tpu-kv-canary-fused-pool && $GA checkout -q tpu-kv-canary-fused-pool && echo '   host a on' \$($GA log -1 --format=%h)"
git checkout -q --detach tpu-kv-canary-fused-pool && log "local on $(git log -1 --format=%h)"
log "=== Qwen3-VL Config A with --kv-canary log: expect fast refusal"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
t0=$(date +%s); env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 900 bash $S/pd.sh start $HOME/pd_disagg/recipes/qwen3-vl_A_128k_canary.json canary_refuse_qwen 2>&1 | grep -E 'ready|NOT' ; echo "   launcher returned after $(( $(date +%s)-t0 ))s"
echo "   prefill (host a): $($SSH_A "sudo -n grep -oE 'kv-canary: the canary runtime is not available on TPU[^\"]*' /home/alisachen_google_com/pd_disagg/results/runs/canary_refuse_qwen/prefill/server.log | head -n 1")"
echo "   decode: $(grep -oE 'kv-canary: the canary runtime is not available on TPU[^"]*' $R/runs/canary_refuse_qwen/decode/server.log | head -n 1)"
echo "   segfaults: prefill $($SSH_A "sudo -n grep -cE 'exit code: -11|Segmentation' /home/alisachen_google_com/pd_disagg/results/runs/canary_refuse_qwen/prefill/server.log") decode $(grep -cE 'exit code: -11|Segmentation' $R/runs/canary_refuse_qwen/decode/server.log)"
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase canary refuse done"
