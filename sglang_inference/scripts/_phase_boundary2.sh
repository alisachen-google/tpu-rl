#!/usr/bin/env bash
# Boundary-case phase (gaps from the coverage review): (1) F-12 stall with 300 s deadlines: does it recover? (2) Config A: ISL+max_new_tokens
# over context, empty prompt, duplicate rid, abort in the transfer window, abort storm; (3) small decode pool: retraction / prealloc pressure.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; P=$HOME/pd_disagg/prompts/gpt-oss; OUT=$R/boundary2; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
log(){ echo "$(date -u +%FT%TZ) [boundary2] $*"; }
state(){ for u in $PRE $DEC; do echo "   $u $(curl -s -m 10 $u/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_used_tokens|num_retracted_reqs|num_prefill_inflight_queue_reqs|num_decode_prealloc_queue_reqs)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' ')"; done; }
start(){ bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $1 $2 2>&1 | tee $R/runs/$2.driver.log | grep -E 'ready|NOT'; curl -sf -m 30 $LB/health_generate >/dev/null; }

log "=== (1) F-12 with 300 s deadlines: single-slot 128k recipe, BOOTSTRAP/WAITING timeout 300"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k_mr1_t300.json A_128k_mr1_t300; then
  t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $P/8192.jsonl --url $LB --out $OUT/f12_t300.jsonl --max-new-tokens 64 --ignore-eos --concurrency 1 --timeout 720 --stream > $OUT/f12_t300.log 2>&1 &
  cpid=$!; for i in $(seq 1 14); do sleep 30; kill -0 $cpid 2>/dev/null || break; echo "   t+$(( $(date +%s)-t0 ))s $(curl -s -m 5 $PRE/metrics | grep -E '^sglang:num_prefill_inflight_queue_reqs' | sed -E 's/\{[^}]*\}//') decode_running=$(curl -s -m 5 $DEC/metrics | grep -E '^sglang:num_running_reqs' | sed -E 's/\{[^}]*\}//' | awk '{print $2}')"; done; wait $cpid
  echo "   client: $(grep -E 'tokens=|ERROR' $OUT/f12_t300.log | cut -c1-160 | tr '\n' ' ')  elapsed=$(( $(date +%s)-t0 ))s"
  echo "   deadline lines: prefill=$($SSH_A "sudo -n grep -cE 'timed out|fail_decode_silence|Request .* timed out' /home/alisachen_google_com/pd_disagg/results/runs/A_128k_mr1_t300/prefill/server.log") decode=$(grep -cE 'timed out|handshake failed|Failed' $R/runs/A_128k_mr1_t300/decode/server.log)"
  $SSH_A "sudo -n grep -E 'timed out|transfer failed|bootstrap failed' /home/alisachen_google_com/pd_disagg/results/runs/A_128k_mr1_t300/prefill/server.log | tail -n 2 | cut -c1-200" | sed 's/^/   prefill: /'; grep -E 'timed out|handshake failed' $R/runs/A_128k_mr1_t300/decode/server.log | tail -n 2 | cut -c1-200 | sed 's/^/   decode: /'
  state
else log "(1) pair did not come up"; fi

log "=== (2) Config A boundary cases"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k.json A_128k_boundary2; then
  T=$(python3 -c "import json; print(json.dumps(json.loads(open('$P/131072.jsonl').readline())['text']))")
  log "(2a) ISL 131072 + max_new_tokens 20000 (> context 140288): expect 400 or clean truncation"
  echo "$T" | python3 -c "import json,sys; print(json.dumps({'text': json.loads(sys.stdin.read()), 'sampling_params': {'temperature':0,'max_new_tokens':20000,'ignore_eos':True}}))" > $OUT/req_over.json
  curl -s -m 900 $LB/generate -H 'Content-Type: application/json' -d @$OUT/req_over.json -o $OUT/resp_over.json -w "   HTTP %{http_code} in %{time_total}s\n"; python3 -c "import json; d=json.load(open('$OUT/resp_over.json')); m=d.get('meta_info',{}); print('   ', {k:m.get(k) for k in ('completion_tokens','finish_reason','prompt_tokens')} if isinstance(d,dict) and 'meta_info' in d else str(d)[:200])"
  log "(2b) ISL 131072 + max_new_tokens 9000 (fits exactly under context): expect 200"
  echo "$T" | python3 -c "import json,sys; print(json.dumps({'text': json.loads(sys.stdin.read()), 'sampling_params': {'temperature':0,'max_new_tokens':9000,'ignore_eos':True}}))" > $OUT/req_fit.json
  curl -s -m 900 $LB/generate -H 'Content-Type: application/json' -d @$OUT/req_fit.json -o $OUT/resp_fit.json -w "   HTTP %{http_code} in %{time_total}s\n"; python3 -c "import json; d=json.load(open('$OUT/resp_fit.json')); m=d.get('meta_info',{}); print('   ', {k:m.get(k) for k in ('completion_tokens','finish_reason','prompt_tokens')} if isinstance(d,dict) and 'meta_info' in d else str(d)[:200])"
  state
  log "(2c) empty / whitespace / empty input_ids prompts"
  for body in '{"text":"","sampling_params":{"max_new_tokens":8}}' '{"text":" ","sampling_params":{"max_new_tokens":8}}' '{"input_ids":[],"sampling_params":{"max_new_tokens":8}}'; do curl -s -m 120 $LB/generate -H 'Content-Type: application/json' -d "$body" -o /dev/null -w "   $body -> HTTP %{http_code} in %{time_total}s\n"; done
  echo "   health: prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
  log "(2d) duplicate rid sent twice concurrently"
  for i in 1 2; do curl -s -m 120 $LB/generate -H 'Content-Type: application/json' -d '{"text":"Count to ten.","rid":"dup-rid-boundary2","sampling_params":{"max_new_tokens":32}}' -o $OUT/dup_$i.json -w "   dup $i -> HTTP %{http_code} in %{time_total}s\n" & done; wait; head -c 160 $OUT/dup_1.json; echo; head -c 160 $OUT/dup_2.json; echo
  sleep 5; state
  log "(2e) abort in the transfer window: 131k prompt aborted at 6.4 s (TTFT ≈ 6.6 s), 3 times"
  $PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase prefill --abort-after 6.4 --repeat 3 --server-info $PRE --server-info $DEC --settle 60 --report $OUT/abort_transfer.json > $OUT/abort_transfer.txt 2>&1; grep -E 'PASS|FAIL|problem' $OUT/abort_transfer.txt | tail -n 2 | cut -c1-200 | sed 's/^/   /'
  echo "   health: prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
  if curl -sf -m 5 $PRE/health >/dev/null; then
    log "(2f) abort storm: 40 streaming 8k requests, 20 concurrent, aborted 3 s in"
    $PY $S/abort_test.py --url $LB --prompt-file $P/8192.jsonl --phase decode --abort-after 3 --repeat 40 --concurrent 20 --server-info $PRE --server-info $DEC --settle 90 --report $OUT/abort_storm.json > $OUT/abort_storm.txt 2>&1; grep -E 'PASS|FAIL|problem' $OUT/abort_storm.txt | tail -n 2 | cut -c1-200 | sed 's/^/   /'
    state
  else log "(2f) skipped: prefill down after (2e)"; $SSH_A "sudo -n grep -E 'Scheduler hit an exception|Error:' /home/alisachen_google_com/pd_disagg/results/runs/A_128k_boundary2/prefill/server.log | tail -n 2 | cut -c1-200" | sed 's/^/   prefill: /'; fi
else log "(2) pair did not come up"; fi

log "=== (3) small decode pool (262,144 tokens, cpu_tensor retraction backup): 4 × 131k prompts, 1,024 tokens, c=4"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k_smallpool.json A_128k_smallpool; then
  python3 - $P/131072.jsonl $OUT/prompts4.jsonl <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
with open(sys.argv[2],'w') as f:
    for i in range(4):
        r=dict(rows[i%len(rows)]); r['id']=f"{r['id']}_dup{i}"; f.write(json.dumps(r)+'\n')
PY
  t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $OUT/prompts4.jsonl --url $LB --out $OUT/smallpool_out.jsonl --max-new-tokens 1024 --ignore-eos --concurrency 4 --timeout 1200 --stream > $OUT/smallpool.log 2>&1 &
  cpid=$!; for i in $(seq 1 40); do sleep 15; kill -0 $cpid 2>/dev/null || break; done; wait $cpid
  echo "   completed: $(grep -cE 'tokens=1024' $OUT/smallpool.log) of 4 in $(( $(date +%s)-t0 ))s; errors: $(grep -cE 'ERROR' $OUT/smallpool.log)"; grep -E 'ERROR' $OUT/smallpool.log | head -n 2 | cut -c1-160 | sed 's/^/   /'
  echo "   decode retraction/pause lines: $(grep -ciE 'retract' $R/runs/A_128k_smallpool/decode/server.log)  metrics: $(curl -s -m 10 $DEC/metrics | grep -E '^sglang:(num_retracted_reqs|num_paused_reqs)' | sed -E 's/\{[^}]*\}//' | tr '\n' ' ')"
  grep -iE 'retract' $R/runs/A_128k_smallpool/decode/server.log | grep -vE 'server_args|UserWarning' | tail -n 3 | cut -c1-200 | sed 's/^/   decode: /'
  $PY $S/needle_check.py --prompts $OUT/prompts4.jsonl --out $OUT/smallpool_out.jsonl 2>&1 | tail -n 1 | sed 's/^/   /'
  state
else log "(3) pair did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; log "phase boundary2 done"
