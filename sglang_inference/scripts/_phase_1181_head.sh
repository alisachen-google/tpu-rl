#!/usr/bin/env bash
# PR #1181 current head (ae27ba40a5) on both hosts: the F-11 failure cases. (1) monolithic trigger 1 (F-12-free), (2) pair trigger 2 small decode pool, (3) pair trigger 1 1M shape.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; P=$HOME/pd_disagg/prompts/gpt-oss; OUT=$R/fix_validate_1181_head; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
BR=pd-swa-chunk-cap-deadlock
log(){ echo "$(date -u +%FT%TZ) [1181-head] $*"; }
health(){ echo "   health: prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"; }
start(){ bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $1 $2 2>&1 | grep -E 'ready|NOT'; curl -sf -m 30 $LB/health_generate >/dev/null; }
swa_pool(){ $SSH_A "sudo -n grep -E 'SWA KV Cache is allocated' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log | head -n 1 | sed -E 's/.*#tokens: ([0-9]+).*/\1/'"; }
trail(){ $SSH_A "sudo -n grep -E 'Prefill batch' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log | sed -E 's/.*#new-token: ([0-9]+).*swa token usage: ([0-9.]+).*/\1@\2/' | tail -n 5 | tr '\n' ' '"; }
excs(){ echo "   scheduler exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception|AssertionError' $R/runs/$1/decode/server.log)"; }

cd $HOME/sglang-torchtpu && git checkout -q main && git fetch -q origin $BR && git bundle create /tmp/p1181b.bundle origin/$BR 2>/dev/null
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/p1181b.bundle sa_101411256518727260534@10.128.15.217:/tmp/p1181b.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/p1181b.bundle; $GA fetch -q /tmp/p1181b.bundle +origin/$BR:$BR && $GA checkout -q --detach $BR && echo '   host a on' \$($GA log -1 --format=%h)"
git checkout -q --detach origin/$BR && log "local on $(git log -1 --format=%h) ($(git log -1 --format=%s | cut -c1-60))"

log "=== (1) F-11 trigger 1, monolithic (no F-12): gpt-oss 1M shape, radix off, 1 slot, host b; 256k prompt (main stalls at chunk 3)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_1m_mr1_noradix.json G_1m_mr1_1181head b 2>&1 | grep -E 'ready|NOT'
L=$(ls $R/runs/G_1m_mr1_1181head/*/server.log $R/runs/G_1m_mr1_1181head/server.log 2>/dev/null | head -n 1)
echo "   SWA pool tokens: $(grep -E 'SWA KV Cache is allocated' $L | head -n 1 | sed -E 's/.*#tokens: ([0-9]+).*/\1/') (main: 16384 -> stall; fix: 16896)"
t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $P/262144.jsonl --url $DEC --out $OUT/mono_262144.jsonl --max-new-tokens 32 --ignore-eos --concurrency 1 --timeout 1200 --stream > $OUT/mono_262144.log 2>&1
echo "   262144: $(grep -E 'tokens=|ERROR' $OUT/mono_262144.log | head -n 1 | cut -c1-150) ($(( $(date +%s)-t0 ))s)"
echo "   chunk trail: $(grep -E 'Prefill batch' $L | sed -E 's/.*#new-token: ([0-9]+).*swa token usage: ([0-9.]+).*/\1@\2/' | tail -n 5 | tr '\n' ' ')"
echo "   health: $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health); exceptions: $(grep -cE 'Scheduler hit an exception|AssertionError' $L)"

log "=== (2) F-11 trigger 2, pair: decode pool 262,144, 4 x 131k prompts, 1,024 tokens, c=4 (main: 1 of 4, pair wedged)"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k_smallpool.json A_smallpool_1181head; then
  echo "   prefill SWA pool tokens: $(swa_pool A_smallpool_1181head)"
  t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $R/boundary2/prompts4.jsonl --url $LB --out $OUT/smallpool_out.jsonl --max-new-tokens 1024 --ignore-eos --concurrency 4 --timeout 900 --stream > $OUT/smallpool.log 2>&1
  echo "   completed: $(grep -cE 'tokens=1024' $OUT/smallpool.log) of 4 in $(( $(date +%s)-t0 ))s; errors: $(grep -cE 'ERROR' $OUT/smallpool.log)"
  $PY $S/needle_check.py --prompts $R/boundary2/prompts4.jsonl --out $OUT/smallpool_out.jsonl 2>&1 | tail -n 1 | sed 's/^/   /'
  echo "   chunk trail: $(trail A_smallpool_1181head)"; health; excs A_smallpool_1181head
else log "(2) pair did not come up"; fi

log "=== (3) F-11 trigger 1, pair: 1 slot, 1M shape; 256k and 1M prompts (main: prefill never past chunk 3; e2e limited by F-12 on 1 slot)"
if start $HOME/pd_disagg/recipes/gpt-oss_A_1m.json A_1m_1181head; then
  echo "   prefill SWA pool tokens: $(swa_pool A_1m_1181head) (main: 16384)"
  for f in 262144 999999; do
    t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $P/$f.jsonl --url $LB --out $OUT/1m_$f.jsonl --max-new-tokens 32 --ignore-eos --concurrency 1 --timeout 1500 --stream > $OUT/1m_$f.log 2>&1
    echo "   $f: $(grep -E 'tokens=|ERROR' $OUT/1m_$f.log | head -n 1 | cut -c1-140) ($(( $(date +%s)-t0 ))s); prefill chunks: $($SSH_A "sudo -n grep -cE 'Prefill batch' /home/alisachen_google_com/pd_disagg/results/runs/A_1m_1181head/prefill/server.log"); last: $(trail A_1m_1181head | awk '{print $NF}')"; health
  done
  excs A_1m_1181head
else log "(3) pair did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase 1181 head done"
