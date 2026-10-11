#!/bin/bash
# Re-validate #1191 after the redesign as a TPU runtime patch (sglang_tpu/pd_decode_queue_limit.py).
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; DEC=http://10.128.0.90:30000; OUT=$R/fix_validate_1191_patch; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
BR=pd-queue-limit-prefill
log(){ echo "$(date -u +%FT%TZ) [1191p] $*"; }
cd $HOME/sglang-torchtpu && git checkout -q main && git fetch -q origin $BR && git bundle create /tmp/q1191p.bundle origin/$BR 2>/dev/null
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/q1191p.bundle sa_101411256518727260534@10.128.15.217:/tmp/q1191p.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/q1191p.bundle; $GA fetch -q /tmp/q1191p.bundle +origin/$BR:$BR && $GA checkout -q --detach $BR && $GA log -1 --format='prefill host on %h'"
git checkout -q --detach origin/$BR && log "decode host on $(git log -1 --format=%h)"
run_case(){ # $1 mode  $2 recipe  $3 name
  log "=== $1: recipe $2"
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $HOME/pd_disagg/recipes/$2 $3 2>&1 | grep -E 'ready|NOT'
  if curl -sf -m 30 $LB/health_generate >/dev/null; then
    $PY $S/queue_limit_probe.py --url $LB --decode $DEC --mode $1 --slots 8 > $OUT/$1.txt 2>&1; sed 's/^/   /' $OUT/$1.txt | cut -c1-260
    echo "   patch installed on decode: $(grep -rlE 'pd_decode_queue_limit' $R/runs/$3/ 2>/dev/null | wc -l) log(s) mention it; scheduler exceptions: decode $(grep -rhcE 'Scheduler hit an exception|AssertionError|Traceback' $R/runs/$3/ 2>/dev/null | awk '{s+=$1} END{print s+0}'), prefill $($SSH_A "sudo -n grep -rhcE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/$3/ 2>/dev/null | awk '{s+=\$1} END{print s+0}'")"
  else echo "   server not ready, skipped"; fi
}
run_case base   gpt-oss_C_128k_ps_q.json      p1191_base
run_case low    gpt-oss_C_128k_ps_q_low.json  p1191_low
run_case q0     gpt-oss_C_128k_ps_q0.json     p1191_q0
run_case nops   gpt-oss_C_128k_q_nops.json    p1191_nops
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main; log "phase 1191p done"
