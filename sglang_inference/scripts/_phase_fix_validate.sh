#!/usr/bin/env bash
# Live validation of the F-8 and F-11 fix branches (combined branch pd-fixes-validate deployed on both hosts).
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; P=$HOME/pd_disagg/prompts/gpt-oss; OUT=$R/fix_validate; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
log(){ echo "$(date -u +%FT%TZ) [fixval] $*"; }
health(){ echo "   health: prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"; }
state(){ for u in $PRE $DEC; do echo "   $u $(curl -s -m 10 $u/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_used_tokens|num_prefill_inflight_queue_reqs|num_decode_prealloc_queue_reqs)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' ')"; done; }
start(){ bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $1 $2 2>&1 | tee $R/runs/$2.driver.log | grep -E 'ready|NOT'; curl -sf -m 30 $LB/health_generate >/dev/null; }
swa_pool(){ $SSH_A "sudo -n grep -E 'SWA KV Cache is allocated' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log | head -n 1 | sed -E 's/.*#tokens: ([0-9]+).*/\1/'"; }

log "=== (1) F-8: aborts during chunked prefill on Config A (radix off), 131k prompt, 3 at 6.4 s then 3 at 2.0 s"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k.json A_128k_fixval; then
  echo "   prefill SWA pool tokens: $(swa_pool A_128k_fixval) (was 19456 before the fix)"
  for t in 6.4 2.0; do
    $PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase prefill --abort-after $t --repeat 3 --server-info $PRE --server-info $DEC --settle 45 --report $OUT/f8_abort_$t.json > $OUT/f8_abort_$t.txt 2>&1
    grep -E 'PASS|FAIL' $OUT/f8_abort_$t.txt | tail -n 1 | cut -c1-200 | sed "s/^/   abort@$t: /"; health
  done
  echo "   prefill scheduler exceptions: $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/A_128k_fixval/prefill/server.log")"
  log "(1b) follow-up correctness: 8k golden prompts still served"
  $PY $S/golden_diff.py generate --prompts $P/8192.jsonl --url $LB --out $OUT/f8_followup.jsonl --max-new-tokens 32 --ignore-eos --concurrency 1 --timeout 300 > $OUT/f8_followup.log 2>&1; echo "   8k prompts completed: $(grep -cE 'tokens=32' $OUT/f8_followup.log)"; state
else log "(1) pair did not come up"; fi

log "=== (2) F-11 trigger 2: small decode pool (262,144), 4 x 131k prompts, 1,024 tokens, c=4 (deadlocked before the fix: 1 of 4)"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k_smallpool.json A_128k_smallpool_fixval; then
  echo "   prefill SWA pool tokens: $(swa_pool A_128k_smallpool_fixval)"
  t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $R/boundary2/prompts4.jsonl --url $LB --out $OUT/smallpool_out.jsonl --max-new-tokens 1024 --ignore-eos --concurrency 4 --timeout 900 --stream > $OUT/smallpool.log 2>&1
  echo "   completed: $(grep -cE 'tokens=1024' $OUT/smallpool.log) of 4 in $(( $(date +%s)-t0 ))s; errors: $(grep -cE 'ERROR' $OUT/smallpool.log)"
  $SSH_A "sudo -n grep -E 'Prefill batch' /home/alisachen_google_com/pd_disagg/results/runs/A_128k_smallpool_fixval/prefill/server.log | sed -E 's/.*#new-token: ([0-9]+).*swa token usage: ([0-9.]+).*#pending-token: ([0-9]+).*/\1@\2/' | tr '\n' ' ' | cut -c1-400" | sed 's/^/   chunks@swa: /'
  $PY $S/needle_check.py --prompts $R/boundary2/prompts4.jsonl --out $OUT/smallpool_out.jsonl 2>&1 | tail -n 1 | sed 's/^/   /'; health; state
else log "(2) pair did not come up"; fi

log "=== (3) F-11 trigger 1: single slot, 1M shape; 256k, 512k, 1M prompts (deadlocked at the third chunk before the fix)"
if start $HOME/pd_disagg/recipes/gpt-oss_A_1m.json A_1m_fixval; then
  echo "   prefill SWA pool tokens: $(swa_pool A_1m_fixval) (was 16384 before the fix)"
  for f in 262144 524288 999999; do
    t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $P/$f.jsonl --url $LB --out $OUT/1m_$f.jsonl --max-new-tokens 32 --ignore-eos --concurrency 1 --timeout 1500 --stream > $OUT/1m_$f.log 2>&1
    echo "   $f: $(grep -E 'tokens=|ERROR' $OUT/1m_$f.log | head -n 1 | cut -c1-140) ($(( $(date +%s)-t0 ))s)"; health
  done
  state
else log "(3) pair did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; log "phase fix validate done"
