#!/usr/bin/env bash
# Re-validate PR #1191 (queue limit on both engines) with the priority probe; waits for the canary-eager phase.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/fix_validate_v2; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [fixv4] $*"; }
while pgrep -f 'canary_refus''e' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main && git bundle create /tmp/q4.bundle pd-queue-limit-prefill 2>/dev/null
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/q4.bundle sa_101411256518727260534@10.128.15.217:/tmp/q4.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/q4.bundle; $GA fetch -q /tmp/q4.bundle pd-queue-limit-prefill:pd-queue-limit-prefill && $GA checkout -q pd-queue-limit-prefill && echo '   host a on' \$($GA log -1 --format=%h)"
git checkout -q --detach pd-queue-limit-prefill && log "local on $(git log -1 --format=%h)"
log "=== #1191 queue limit on the decode (decode-only design): Config C + --max-queued-requests 4 + priority scheduling"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_C_128k_ps_q.json fixv4_C_ps_q 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  cd $S && $PY priority_probe.py --url $LB --decode $DEC --mode q --slots 8 --hold-tokens 3000 > $OUT/v4_ps_q.txt 2>&1; grep -E '^\[D|^\[end|^\[A4-health' $OUT/v4_ps_q.txt | sed 's/^/   /' | cut -c1-230
  echo "   exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/fixv4_C_ps_q/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception' $R/runs/fixv4_C_ps_q/decode/server.log); used tokens decode $(curl -s -m 5 $DEC/metrics | grep -E '^sglang:num_used_tokens' | head -n1 | sed -E 's/.* //'); health prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
  grep -E 'queue is full|higher priority request' $R/runs/fixv4_C_ps_q/decode/server.log | head -n 2 | cut -c1-160 | sed 's/^/   decode log: /'
else log "pair did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase fix validate v4 done"
