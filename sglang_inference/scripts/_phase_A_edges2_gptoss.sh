#!/usr/bin/env bash
# Resume of the Config A gpt-oss edge chain after the F-5 crash. Dangerous repros (F-6 n=2, F-5 prompt logprobs) run LAST.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results/A; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; M=gpt-oss; P=$HOME/pd_disagg/prompts/$M
RUNDIR=$HOME/pd_disagg/results/runs/A_gptoss_128k_run1
log(){ echo "$(date -u +%FT%TZ) $*"; }
state(){ for u in $PRE $DEC; do echo "   $u $(curl -s -m 10 $u/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|full_token_usage|num_used_tokens|num_retracted_reqs|num_decode_transfer_queue_reqs|num_decode_prealloc_queue_reqs|num_prefill_inflight_queue_reqs)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' ')"; done; }
settle(){ sleep ${1:-30}; log "state after settle:"; state; }
until curl -sf -m 5 $PRE/health >/dev/null && curl -sf -m 30 $LB/health_generate >/dev/null; do sleep 20; done; log "prefill + LB back"; state
log "(2b) params matrix without n_2 / return_logprob_top5"
$PY $S/params_matrix.py --a $LB --prompt-file $P/8192.jsonl --cases greedy_baseline,top_k_1,repetition_penalty_1.3,frequency_penalty_0.8,presence_penalty_0.8,min_new_tokens_100_ignore_eos_false,stop_string,stop_token_ids,no_stop_trim,logit_bias,max_new_tokens_1,max_new_tokens_0,skip_special_tokens_false,sampling_seed_42_temp0.7,json_schema,regex,stream_vs_nonstream --report $R/params/$M/params_pd_A2.json --timeout 300 > $R/params/$M/params_pd_A2.txt 2>&1; log "params rc=$?"; tail -n 20 $R/params/$M/params_pd_A2.txt | cut -c1-160
settle 15
log "(3) chat suite"
$PY $S/chat_suite.py --model gpt-oss-120b --kind gptoss --a $LB --report $R/chat/$M/chat_pd_A.json --timeout 600 > $R/chat/$M/chat_pd_A.txt 2>&1; log "chat rc=$?"; tail -n 16 $R/chat/$M/chat_pd_A.txt | cut -c1-160
settle 15
log "(4) concurrency sweep c=1,2,4 at ISL 131072 / OSL 256"
cd $HOME/sglang-torchtpu
for c in 1 2 4; do
  $PY -m sglang.bench_serving --backend sglang --base-url $LB --dataset-name random --random-input-len 131072 --random-output-len 256 --random-range-ratio 1.0 \
     --num-prompts $((c*2)) --max-concurrency $c --request-rate inf --disable-ignore-eos --extra-request-body '{"temperature":0}' \
     --output-file $R/conc/$M/random_131072_256_c$c.jsonl > $R/conc/$M/random_131072_256_c$c.txt 2>&1
  grep -E 'Successful|Failed|Mean TTFT|P99 TTFT|Mean ITL|P99 ITL|Output token throughput' $R/conc/$M/random_131072_256_c$c.txt | sed "s/^/   c=$c /"
  settle 15
done
log "(5) retraction test: 8 x 131072 at c=8, 1024 new tokens"
$PY - $P/131072.jsonl $R/retract/$M/prompts8.jsonl <<'EOF'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
with open(sys.argv[2],'w') as f:
    for i in range(8):
        r=dict(rows[i%len(rows)]); r['id']=f"{r['id']}_dup{i}"; f.write(json.dumps(r)+'\n')
EOF
$PY $S/golden_diff.py generate --prompts $R/retract/$M/prompts8.jsonl --url $LB --out $R/retract/$M/out8.jsonl --max-new-tokens 1024 --ignore-eos --concurrency 8 --timeout 3600 --stream > $R/retract/$M/generate.log 2>&1
grep -E 'errors' $R/retract/$M/generate.log | cut -c1-160
$PY - $P/131072.jsonl $R/retract/$M/out8.jsonl $R/tc1/$M/131072.jsonl <<'EOF' | tee $R/retract/$M/check.txt
import json,sys
pr={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[1])}
outs=[json.loads(l) for l in open(sys.argv[2])]
c1={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[3])}
groups={}
for o in outs: groups.setdefault(o['id'].rsplit('_dup',1)[0],[]).append(o)
for base,g in groups.items():
    nd=list(pr[base]['needles'].values()); texts=[x.get('text') or '' for x in g]
    found=[sum(w in t for w in nd) for t in texts]
    same_as_c1=[t==(c1.get(base,{}).get('text') or '') for t in texts]
    print(f"{base}: copies={len(g)} needles/copy={found} distinct_texts={len(set(texts))} identical_to_c1_run={same_as_c1} tokens={[x.get('completion_tokens') for x in g]} errors={[x.get('error') for x in g if x.get('error')]}")
EOF
echo "   decode retraction gauge: $(curl -s -m 10 $DEC/metrics | grep -E '^sglang:num_retracted_reqs' | head -1)"; grep -ciE 'retract' $RUNDIR/decode/server.log | sed 's/^/   decode log retraction lines: /'
settle 60
log "(6) client aborts: prefill phase, then decode phase"
$PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase prefill --abort-after 2 --repeat 3 --server-info $PRE --server-info $DEC --settle 60 --report $R/abort/$M/abort_prefill.json > $R/abort/$M/abort_prefill.txt 2>&1; log "abort prefill rc=$?"; tail -n 6 $R/abort/$M/abort_prefill.txt | cut -c1-160
$PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase decode --abort-after 5 --repeat 3 --concurrent 2 --server-info $PRE --server-info $DEC --settle 60 --report $R/abort/$M/abort_decode.json > $R/abort/$M/abort_decode.txt 2>&1; log "abort decode rc=$?"; tail -n 6 $R/abort/$M/abort_decode.txt | cut -c1-160
settle 30
log "(8) GSM8K-100 through the LB"
cd $HOME/sglang-torchtpu && $PY -m sglang.test.run_eval --base-url $LB --model gpt-oss-120b --eval-name gsm8k --num-examples 100 --num-threads 4 --api chat --max-tokens 2048 --temperature 0 > $R/gsm8k_pd_A.log 2>&1; grep -E 'gsm8k_score|Error|Traceback' $R/gsm8k_pd_A.log | head -3
settle 30
log "(7) compile audit"
bash $S/compile_audit.sh $RUNDIR/decode/server.log decode_A | tee $R/compile_audit_decode_A.txt
log "(9) F-6 repro: single n=2 request, 120 s timeout, with state before/after"
state
curl -s -m 120 $LB/generate -H 'Content-Type: application/json' -d '{"text":"The three primary colors are","sampling_params":{"temperature":0,"max_new_tokens":32,"n":2}}' -o $R/params/$M/n2_repro.json -w "   n=2 HTTP %{http_code} in %{time_total}s\n"; head -c 400 $R/params/$M/n2_repro.json; echo
sleep 20; state
log "(9b) n=2 with temperature 0.7 (sampling path)"
curl -s -m 120 $LB/generate -H 'Content-Type: application/json' -d '{"text":"The three primary colors are","sampling_params":{"temperature":0.7,"max_new_tokens":32,"n":2}}' -o $R/params/$M/n2_repro_t07.json -w "   n=2 t=0.7 HTTP %{http_code} in %{time_total}s\n"; head -c 300 $R/params/$M/n2_repro_t07.json; echo
sleep 20; state
log "(10) F-5 repro LAST (expected to kill the prefill engine): prompt logprobs request"
curl -s -m 120 $LB/generate -H 'Content-Type: application/json' -d '{"text":"hi there","sampling_params":{"max_new_tokens":4},"return_logprob":true,"logprob_start_len":0}' -o $R/params/$M/f5_repro.json -w "   prompt-logprob HTTP %{http_code} in %{time_total}s\n"; head -c 300 $R/params/$M/f5_repro.json; echo
sleep 15; curl -s -m 5 -o /dev/null -w "   prefill /health after F-5 repro: %{http_code}\n" $PRE/health; state
log "phase A edges done"
