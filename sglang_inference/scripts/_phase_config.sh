#!/usr/bin/env bash
# Generic PD configuration test: _phase_config.sh <model: gpt-oss|qwen3-vl> <cfg: B|C|D|E> <golden_dir (c=1 goldens)>
# Starts the pair from recipes/<model>_<cfg>_128k.json, runs the TC battery, archives both engine logs, stops the pair.
# Each step is independent; a failed step is logged and the next one runs. Prefill-phase abort runs LAST (kills the prefill in Config A).
M=${1:?model}; CFG=${2:?cfg}; GOLD=${3:?golden dir}
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; P=$HOME/pd_disagg/prompts/$M
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
RUN=${CFG}_${M}_128k; OUT=$R/$CFG/$M; mkdir -p $OUT; REC=$HOME/pd_disagg/recipes/${M}_${CFG}_128k.json
case $M in gpt-oss) MODEL_NAME=gpt-oss-120b; KIND=gptoss; GSM_TOK=2048; GSM_THR=4;; qwen3-vl) MODEL_NAME=qwen3-vl-30b; KIND=qwen; GSM_TOK=1024; GSM_THR=8;; esac
log(){ echo "$(date -u +%FT%TZ) [$CFG/$M] $*"; }
state(){ for u in $PRE $DEC; do echo "   $u $(curl -s -m 10 $u/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_used_tokens|swa_token_usage)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' ')"; done; }
prefill_log(){ $SSH_A "sudo -n cat /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log" 2>/dev/null; }
healthy(){ curl -sf -m 5 $PRE/health >/dev/null && curl -sf -m 5 $DEC/health >/dev/null; }

log "=== start: recipe $REC, goldens $GOLD"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; bash $S/tpu_free.sh || { log "TPU not free; skipping config"; exit 1; }
t0=$(date +%s)
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $REC $RUN 2>&1 | tee $R/runs/$RUN.driver.log | grep -E 'ready|NOT'
if ! curl -sf -m 30 $LB/health_generate >/dev/null; then
  log "PAIR DID NOT COME UP (after $(( $(date +%s) - t0 )) s). Last error lines:"
  echo "--- decode:"; grep -E 'Error|error|Exception|died|Check failed|CHECK|Traceback|exited' $R/runs/$RUN/decode/server.log 2>/dev/null | grep -vE 'UserWarning|GET /' | tail -n 8 | cut -c1-220
  echo "--- prefill:"; prefill_log | grep -E 'Error|error|Exception|died|Check failed|CHECK|Traceback|exited' | grep -vE 'UserWarning|GET /' | tail -n 8 | cut -c1-220
  prefill_log | gzip -c > $OUT/prefill_server_startfail.log.gz; gzip -c $R/runs/$RUN/decode/server.log > $OUT/decode_server_startfail.log.gz 2>/dev/null
  bash $S/pd.sh stop >/dev/null 2>&1; log "=== config $CFG/$M done (start failure)"; exit 2
fi
log "pair up in $(( $(date +%s) - t0 )) s"; curl -s -m 10 $PRE/get_server_info > $OUT/server_info_prefill.json; curl -s -m 10 $DEC/get_server_info > $OUT/server_info_decode.json
nohup $PY $S/pd_watchdog.py --server prefill=$PRE --server decode=$DEC --log decode=$R/runs/$RUN/decode/server.log --interval 10 --hang-secs 300 --out $OUT/watchdog.csv > $OUT/watchdog.out 2>&1 &
WD=$!
state

log "(1) TC-1: c=1 greedy, 1024 tokens, 8k/32k/128k vs goldens in $GOLD"
for L in 8192 32768 131072; do
  $PY $S/golden_diff.py generate --prompts $P/$L.jsonl --url $LB --out $OUT/tc1_$L.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 --timeout 1800 --stream > $OUT/tc1_generate_$L.log 2>&1
  grep -E 'tokens=|ERROR' $OUT/tc1_generate_$L.log | cut -c1-160 | sed 's/^/   /'
  [ -f $GOLD/$L.jsonl ] && $PY $S/golden_diff.py diff --a $GOLD/$L.jsonl --b $OUT/tc1_$L.jsonl --report $OUT/tc1_diff_$L.json --expect-tokens 1024 2>&1 | grep -E '^isl|ids,' | sed 's/^/   vs golden /'
  $PY $S/needle_check.py --prompts $P/$L.jsonl --out $OUT/tc1_$L.jsonl --report $OUT/needles_$L.json 2>&1 | tail -n 2 | sed 's/^/   needles /'
done
state

log "(2) tiny boundary sweep (160 tokens, c=1) + needles"
$PY $S/golden_diff.py generate --prompts $P/tiny.jsonl --url $LB --out $OUT/tiny.jsonl --max-new-tokens 160 --ignore-eos --return-logprob --concurrency 1 --timeout 600 > $OUT/tiny_generate.log 2>&1
$PY - $P/tiny.jsonl $OUT/tiny.jsonl <<'EOF'
import json,sys
pr={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[1])}; out={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[2])}
bad=[k for k,p in pr.items() if (o:=out.get(k,{})) is not None and ((p.get('needle') and p['needle'] not in (o.get('text') or '')) or o.get('error') or o.get('completion_tokens')!=160)]
print(f"   tiny: {len(pr)} prompts, {len(bad)} problems {bad[:8]}")
EOF

if [[ $CFG =~ [BCE] ]]; then
  log "(3) prefix-cache reuse: 32k prompts twice at c=1 (64 tokens); second pass should report cached_tokens>0 and identical text"
  $PY $S/golden_diff.py generate --prompts $P/32768.jsonl --url $LB --out $OUT/reuse_pass1.jsonl --max-new-tokens 64 --ignore-eos --return-logprob --concurrency 1 --timeout 600 > $OUT/reuse1.log 2>&1
  $PY $S/golden_diff.py generate --prompts $P/32768.jsonl --url $LB --out $OUT/reuse_pass2.jsonl --max-new-tokens 64 --ignore-eos --return-logprob --concurrency 1 --timeout 600 > $OUT/reuse2.log 2>&1
  $PY - $OUT/reuse_pass1.jsonl $OUT/reuse_pass2.jsonl <<'EOF'
import json,sys
a={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[1])}; b={json.loads(l)['id']:json.loads(l) for l in open(sys.argv[2])}
for k in a:
    x,y=a[k],b.get(k,{})
    print(f"   {k}: pass1 cached={x.get('cached_tokens')} ttft={x.get('ttft_ms')}ms | pass2 cached={y.get('cached_tokens')} ttft={y.get('ttft_ms')}ms | identical_text={x.get('text')==y.get('text')} first_logprob_delta={abs((x.get('first_token_logprob') or 0)-(y.get('first_token_logprob') or 0)):.4f}")
EOF
  echo "   prefill cache metrics: $(curl -s -m 10 $PRE/metrics | grep -E '^sglang:(cache_hit_rate|num_cached_tokens|prefix_hit)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' ')"
fi
if [[ $CFG =~ [CDE] ]]; then
  log "(4) HiCache: host-cache lines in the prefill log so far"
  prefill_log | grep -iE 'hicache|host cache|write.through|offload' | grep -vE 'server_args|UserWarning' | tail -n 6 | cut -c1-200 | sed 's/^/   /'
fi
if [[ $CFG =~ [DE] ]]; then
  log "(5) EAGLE3: acceptance length and spec lines"
  for u in $PRE $DEC; do echo "   $u avg_spec_accept_length=$(curl -s -m 10 $u/get_server_info | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("internal_states",[{}])[0].get("avg_spec_accept_length"))' 2>/dev/null)"; done
  grep -iE 'accept|spec' $R/runs/$RUN/decode/server.log | grep -vE 'server_args|UserWarning|speculative_' | tail -n 4 | cut -c1-200 | sed 's/^/   decode /'
fi
state

log "(6) sampling-params matrix (n_2 excluded: F-6)"
$PY $S/params_matrix.py --a $LB --prompt-file $P/8192.jsonl --cases greedy_baseline,top_k_1,repetition_penalty_1.3,frequency_penalty_0.8,presence_penalty_0.8,min_new_tokens_100_ignore_eos_false,stop_string,stop_token_ids,no_stop_trim,logit_bias,return_logprob_top5,max_new_tokens_1,max_new_tokens_0,skip_special_tokens_false,sampling_seed_42_temp0.7,json_schema,regex,stream_vs_nonstream --report $OUT/params.json --timeout 300 > $OUT/params.txt 2>&1; log "params rc=$?"; grep -E 'cases|FAIL|PASS' $OUT/params.txt | tail -n 3 | cut -c1-160 | sed 's/^/   /'
healthy || log "!! an engine is down after the params matrix"

log "(7) chat suite"
$PY $S/chat_suite.py --model $MODEL_NAME --kind $KIND --a $LB --report $OUT/chat.json --timeout 600 > $OUT/chat.txt 2>&1; log "chat rc=$?"; grep -E 'cases|FAIL' $OUT/chat.txt | tail -n 3 | cut -c1-160 | sed 's/^/   /'
if [[ $M == qwen3-vl ]]; then
  log "(7b) multimodal: 1x448 and 4x448 images"
  for n in 1 4; do $PY $S/mm_request.py --url $LB --model $MODEL_NAME --num-images $n --resolution 448x448 --max-tokens 48 --temperature 0 2>&1 | grep -E 'status=' | cut -c1-160 | sed "s/^/   ${n}x448: /"; done
fi

log "(8) concurrency sweep c=1,2,4 at ISL 131072 / OSL 256"
cd $HOME/sglang-torchtpu
for c in 1 2 4; do
  $PY -m sglang.bench_serving --backend sglang --base-url $LB --dataset-name random --random-input-len 131072 --random-output-len 256 --random-range-ratio 1.0 \
     --num-prompts $((c*2)) --max-concurrency $c --request-rate inf --disable-ignore-eos --extra-request-body '{"temperature":0}' \
     --output-file $OUT/conc_c$c.jsonl > $OUT/conc_c$c.txt 2>&1
  grep -E 'Successful|Failed|Mean TTFT|P99 TTFT|Mean ITL|Output token throughput' $OUT/conc_c$c.txt | tr -s ' ' | tr '\n' ';' | cut -c1-240 | sed "s/^/   c=$c /"; echo
done
cd $HOME
state

log "(9) GSM8K-100 via LB"
cd $HOME/sglang-torchtpu && $PY -m sglang.test.run_eval --base-url $LB --model $MODEL_NAME --eval-name gsm8k --num-examples 100 --num-threads $GSM_THR --api chat --max-tokens $GSM_TOK --temperature 0 > $OUT/gsm8k.log 2>&1; grep -E 'gsm8k_score|Traceback' $OUT/gsm8k.log | tail -n 2 | sed 's/^/   /'; cd $HOME

log "(10) decode-phase client aborts (3, two concurrent)"
$PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase decode --abort-after 5 --repeat 3 --concurrent 2 --server-info $PRE --server-info $DEC --settle 60 --report $OUT/abort_decode.json > $OUT/abort_decode.txt 2>&1; log "abort decode rc=$?"; grep -E 'PASS|FAIL|problem' $OUT/abort_decode.txt | tail -n 2 | cut -c1-200 | sed 's/^/   /'
state

log "(11) compile audit (decode, prefill)"
bash $S/compile_audit.sh $R/runs/$RUN/decode/server.log decode_$CFG | head -n 3 | sed 's/^/   /'
prefill_log > $OUT/prefill_server.log; bash $S/compile_audit.sh $OUT/prefill_server.log prefill_$CFG | head -n 3 | sed 's/^/   /'

log "(12) LAST: prefill-phase client abort (crashes the prefill engine in Config A, F-8)"
$PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase prefill --abort-after 2 --repeat 1 --server-info $PRE --server-info $DEC --settle 30 --report $OUT/abort_prefill.json > $OUT/abort_prefill.txt 2>&1; log "abort prefill rc=$?"
sleep 15; echo "   prefill /health after prefill-abort: $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health)   decode: $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
prefill_log | grep -E 'Scheduler hit an exception|AssertionError|Error:' | grep -vE 'UserWarning' | tail -n 3 | cut -c1-200 | sed 's/^/   /'

kill $WD 2>/dev/null
log "archiving logs"; mkdir -p $R/logs/runs/$RUN; prefill_log | gzip -c > $R/logs/runs/$RUN/prefill_server.log.gz; gzip -c $R/runs/$RUN/decode/server.log > $R/logs/runs/$RUN/decode_server.log.gz; rm -f $OUT/prefill_server.log
bash $S/pd.sh stop >/dev/null 2>&1
log "=== config $CFG/$M done"
