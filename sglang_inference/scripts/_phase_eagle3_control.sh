#!/usr/bin/env bash
# EAGLE3 control: monolithic gpt-oss (bf16 KV) with the same speculative flags as Config D. Measures accept length / throughput without P/D.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
U=http://10.128.0.90:30000; OUT=$R/D/gpt-oss/eagle3_control; mkdir -p $OUT; P=$HOME/pd_disagg/prompts/gpt-oss
log(){ echo "$(date -u +%FT%TZ) [eagle3-control] $*"; }
until grep -q 'phase kv integrity done' $R/phase_kv_integrity.log 2>/dev/null; do sleep 120; done; log "kv integrity done"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
RUN=G_gptoss_128k_bf16_eagle3; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_bfloat16_eagle3.json $RUN b > $R/runs/$RUN.driver.log 2>&1
if ! curl -sf -m 5 $U/health >/dev/null; then log "monolithic EAGLE3 server did not come up"; grep -hE 'Error|died|Check failed|Traceback' $R/runs/$RUN/golden/server.log | grep -vE 'UserWarning|freeze_gc' | tail -n 4 | cut -c1-200; bash $S/pd.sh stop; exit 2; fi
log "(1) TC-1 8k vs bf16 golden (identity check of the spec verify path, monolithic)"
$PY $S/golden_diff.py generate --prompts $P/8192.jsonl --url $U --out $OUT/8192.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 --timeout 900 --stream > $OUT/8192.log 2>&1; grep -E 'tokens=' $OUT/8192.log | cut -c1-140 | sed 's/^/   /'
$PY $S/golden_diff.py diff --a $R/golden/gpt-oss_bf16_c1/8192.jsonl --b $OUT/8192.jsonl --report $OUT/diff_8192.json --expect-tokens 1024 2>&1 | grep -E '^isl|ids,' | sed 's/^/   vs bf16 golden /'
echo "   accept after random-text TC-1: $(curl -s -m 10 $U/metrics | grep -E '^sglang:spec_(accept_length|accept_rate)' | sed -E 's/\{[^}]*\}//' | tr '\n' ' ')"
log "(2) GSM8K-100 (natural text): accuracy, throughput and acceptance, monolithic EAGLE3"
cd $HOME/sglang-torchtpu && $PY -m sglang.test.run_eval --base-url $U --model gpt-oss-120b --eval-name gsm8k --num-examples 100 --num-threads 4 --api chat --max-tokens 2048 --temperature 0 > $OUT/gsm8k.log 2>&1; cd $HOME; grep -E 'gsm8k_score|Total latency|Output throughput' $OUT/gsm8k.log | sed 's/^/   /'
echo "   accept after GSM8K: $(curl -s -m 10 $U/metrics | grep -E '^sglang:spec_(accept_length|accept_rate)' | sed -E 's/\{[^}]*\}//' | tr '\n' ' ')  cumulative avg_spec_accept_length=$(curl -s -m 10 $U/get_server_info | python3 -c 'import json,sys; print(json.load(sys.stdin).get("internal_states",[{}])[0].get("avg_spec_accept_length"))')"
grep -E 'Decode batch' $R/runs/$RUN/golden/server.log | grep -oE 'accept len: [0-9.]+' | awk '{print $3}' | sort -n | awk '{a[NR]=$1} END{print "   accept len over all decode batches: n="NR, "min="a[1], "p50="a[int(NR/2)], "max="a[NR]}'
bash $S/pd.sh stop >/dev/null 2>&1; log "phase eagle3 control done"
