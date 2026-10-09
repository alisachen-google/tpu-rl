#!/usr/bin/env bash
# Scenario battery for PR #1181 (F-11 fix) on top of upstream main (which carries the F-8 fix). Deploys the PR head to both hosts via a
# git bundle, runs gpt-oss and Qwen scenarios, restores both checkouts to main. Queued behind _phase_1161.sh.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; P=$HOME/pd_disagg/prompts/gpt-oss; Q=$HOME/pd_disagg/prompts/qwen3-vl; OUT=$R/fix_scenarios; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [scen] $*"; }
health(){ echo "   health: prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"; }
excs(){ echo "   exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception' $R/runs/$1/decode/server.log); recompiles after warmup: prefill $($SSH_A "sudo -n grep -cE 'Recompil|recompile' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log") decode $(grep -cE 'Recompil|recompile' $R/runs/$1/decode/server.log)"; }
swa(){ $SSH_A "sudo -n grep -E 'SWA KV Cache is allocated' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log | tail -n 1 | sed -E 's/.*#tokens: ([0-9]+).*/\1/'"; }
start(){ bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $1 $2 2>&1 | tee $R/runs/$2.driver.log | grep -E 'ready|NOT'; curl -sf -m 30 $LB/health_generate >/dev/null; }
gen(){ # $1 prompts $2 out-tag $3 max_new $4 conc $5 timeout
  t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $1 --url $LB --out $OUT/$2.jsonl --max-new-tokens $3 --ignore-eos --concurrency $4 --timeout $5 --stream > $OUT/$2.log 2>&1
  echo "   $2: completed $(grep -cE 'tokens=' $OUT/$2.log)/$(grep -cE '^\{' $1) errors $(grep -cE 'ERROR' $OUT/$2.log) in $(( $(date +%s)-t0 ))s"; }
while pgrep -f '[_]phase_1161.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main && git branch -f pr1181-check pd-swa-chunk-cap-deadlock && git bundle create /tmp/pr1181.bundle pr1181-check 2>/dev/null && log "PR head: $(git log -1 --format=%h pr1181-check)"
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/pr1181.bundle sa_101411256518727260534@10.128.15.217:/tmp/pr1181.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/pr1181.bundle; $GA fetch -q /tmp/pr1181.bundle pr1181-check:pr1181-check && $GA checkout -q pr1181-check && echo '   host a on' \$($GA log -1 --format=%h)"
git checkout -q pr1181-check && log "local on $(git log -1 --format=%h)"
rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' $HOME/pd_disagg/recipes/ sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/recipes/

log "=== S1 Config A 128k (radix off, 8 slots): correctness vs golden, edges, aborts, 8x131k back-pressure"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k.json scen_A_128k; then
  echo "   SWA pool: $(swa scen_A_128k) (expect 19968)"
  for isl in 8192 131072; do
    gen $P/$isl.jsonl S1_golden_$isl 1024 1 1500
    $PY $S/golden_diff.py diff --a $R/golden/gpt-oss_fp8/$isl.jsonl --b $OUT/S1_golden_$isl.jsonl --report $OUT/S1_golden_$isl.report.json > $OUT/S1_golden_$isl.diff.log 2>&1; echo "   golden diff $isl (monolithic golden vs PR pair, c=1): $(tail -n 1 $OUT/S1_golden_$isl.diff.log | cut -c1-160)"
  done
  gen $P/8192.jsonl S1_8k 64 2 600
  gen $P/131072.jsonl S1_131k_c1 32 1 900
  python3 - $P/131072.jsonl $OUT/p8.jsonl <<'PY'
import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]
with open(sys.argv[2],'w') as f:
    for i in range(8): r=dict(rows[i%len(rows)]); r['id']=f"{r['id']}_x{i}"; f.write(json.dumps(r)+'\n')
PY
  gen $OUT/p8.jsonl S1_131k_c8 256 8 1500
  $PY $S/needle_check.py --prompts $OUT/p8.jsonl --out $OUT/S1_131k_c8.jsonl 2>&1 | tail -n 1 | sed 's/^/   needles: /'
  $PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase prefill --abort-after 3 --repeat 3 --concurrent 2 --server-info $PRE --server-info $DEC --settle 45 --report $OUT/S1_abort.json > $OUT/S1_abort.txt 2>&1; grep -E 'PASS|FAIL' $OUT/S1_abort.txt | tail -n 1 | cut -c1-180 | sed 's/^/   abort x2 concurrent: /'
  excs scen_A_128k; health
else log "S1 pair did not come up"; fi

log "=== S2 small decode pool, heavier: 8 x 131k at c=8 then 16 x 32k at c=16"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k_smallpool.json scen_A_smallpool; then
  gen $OUT/p8.jsonl S2_131k_c8 512 8 1800
  python3 - $P/32768.jsonl $OUT/p16.jsonl <<'PY'
import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]
with open(sys.argv[2],'w') as f:
    for i in range(16): r=dict(rows[i%len(rows)]); r['id']=f"{r['id']}_x{i}"; f.write(json.dumps(r)+'\n')
PY
  gen $OUT/p16.jsonl S2_32k_c16 256 16 1800
  excs scen_A_smallpool; health
else log "S2 pair did not come up"; fi

log "=== S3 1M shape, 4 slots (radix off): 4 x 256k at c=4, then 1M single"
if start $HOME/pd_disagg/recipes/gpt-oss_A_1m_mr4.json scen_A_1m_mr4; then
  echo "   SWA pool: $(swa scen_A_1m_mr4)"
  python3 - $P/262144.jsonl $OUT/p4_256k.jsonl <<'PY'
import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]
with open(sys.argv[2],'w') as f:
    for i in range(4): r=dict(rows[i%len(rows)]); r['id']=f"{r['id']}_x{i}"; f.write(json.dumps(r)+'\n')
PY
  gen $OUT/p4_256k.jsonl S3_256k_c4 64 4 2400
  gen $P/999999.jsonl S3_1m 32 1 2400
  excs scen_A_1m_mr4; health
else log "S3 pair did not come up"; fi

log "=== S4 regression on radix-on configs (fix is a no-op there): B and C 128k, golden + reuse"
for cfg in B C; do
  if start $HOME/pd_disagg/recipes/gpt-oss_${cfg}_128k.json scen_${cfg}_128k; then
    gen $P/131072.jsonl S4_${cfg}_131k 32 1 900; gen $P/131072.jsonl S4_${cfg}_131k_reuse 32 1 900
    echo "   reuse cached_tokens: $(grep -oE 'cached=[0-9]+' $OUT/S4_${cfg}_131k_reuse.log | head -n 1)"; excs scen_${cfg}_128k; health
  else log "S4 $cfg pair did not come up"; fi
done

log "=== S5 Qwen3-VL Config A 128k (no SWA: alignment path not taken; raiden early release applies): text + image"
if start $HOME/pd_disagg/recipes/qwen3-vl_A_128k.json scen_qwen_A; then
  gen $Q/8192.jsonl S5_qwen_8k 1024 1 900
  $PY $S/golden_diff.py diff --a $R/golden/qwen3-vl_fp8/8192.jsonl --b $OUT/S5_qwen_8k.jsonl --report $OUT/S5_qwen_8k.report.json > $OUT/S5_qwen_8k.diff.log 2>&1; echo "   golden diff qwen 8k: $(tail -n 1 $OUT/S5_qwen_8k.diff.log | cut -c1-160)"
  for spec in "1 1920x1080" "4 448x448" "16 448x448"; do set -- $spec; $PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images $1 --resolution $2 --max-tokens 48 --temperature 0 --needle-word plover > $OUT/S5_mm_${1}x$2.log 2>&1; echo "   mm ${1}x$2: $(grep -iE 'needle|status|tokens|error' $OUT/S5_mm_${1}x$2.log | tail -n 1 | cut -c1-160)"; done
  excs scen_qwen_A; health
else log "S5 pair did not come up"; fi

bash $S/pd.sh stop >/dev/null 2>&1
$SSH_A "$GA checkout -q main && echo '   host a restored to' \$($GA log -1 --format=%h)"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase fix scenarios done (local on $(git branch --show-current))"
