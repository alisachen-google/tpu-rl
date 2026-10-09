#!/usr/bin/env bash
# Verify upstream main's chunked-abort fixes (bab49bcec2, 7366de9460, 2645aa14af) against our F-8 reproduction.
# Deploys origin/main to both hosts via a git bundle (no credential leaves this VM), runs the abort tests, restores both checkouts.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; P=$HOME/pd_disagg/prompts/gpt-oss; OUT=$R/fix_validate; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [f8-upstream] $*"; }
while pgrep -f '[_]phase_fix_validate_mono.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main && git branch -f upstream-main-check origin/main && git bundle create /tmp/upstream-main.bundle upstream-main-check 2>/dev/null && log "bundle: $(git log -1 --format=%h upstream-main-check)"
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/upstream-main.bundle sa_101411256518727260534@10.128.15.217:/tmp/upstream-main.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/upstream-main.bundle; $GA fetch -q /tmp/upstream-main.bundle upstream-main-check:upstream-main-check && $GA checkout -q upstream-main-check && echo 'host a on' \$($GA log -1 --format=%h)"
git checkout -q upstream-main-check && log "local on $(git log -1 --format=%h)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k.json A_128k_upstream_f8 2>&1 | tee $R/runs/A_128k_upstream_f8.driver.log | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  for t in 2.0 6.4; do
    $PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase prefill --abort-after $t --repeat 3 --server-info $PRE --server-info $DEC --settle 45 --report $OUT/f8_upstream_abort_$t.json > $OUT/f8_upstream_abort_$t.txt 2>&1
    grep -E 'PASS|FAIL' $OUT/f8_upstream_abort_$t.txt | tail -n 1 | cut -c1-200 | sed "s/^/   abort@$t: /"; echo "   health: prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
  done
  echo "   prefill scheduler exceptions: $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/A_128k_upstream_f8/prefill/server.log")"
else log "pair did not come up"; $SSH_A "sudo -n grep -E 'Error|error' /home/alisachen_google_com/pd_disagg/results/runs/A_128k_upstream_f8/prefill/server.log | grep -vE UserWarning | tail -n 3 | cut -c1-200"; fi
bash $S/pd.sh stop >/dev/null 2>&1
$SSH_A "$GA checkout -q main && echo 'host a restored to' \$($GA log -1 --format=%h)"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase f8 upstream check done (local on $(git branch --show-current))"
