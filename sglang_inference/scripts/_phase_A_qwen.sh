#!/usr/bin/env bash
# After the gpt-oss edge chain: switch to Config A Qwen3-VL (128k shape) and run TC-1 + multimodal edge cases.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; M=qwen3-vl; P=$HOME/pd_disagg/prompts/$M
log(){ echo "$(date -u +%FT%TZ) $*"; }
state(){ for u in $PRE $DEC; do curl -s -m 10 $u/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|full_token_usage|num_used_tokens|num_retracted_reqs)' | awk -v u=$u '{printf "%s %s=%s  ", u, $1, $2} END{print ""}'; done; }
until grep -q 'phase A edges done' $R/A/edges_gptoss.log 2>/dev/null; do sleep 60; done; log "gpt-oss edges done"
bash $S/compile_audit.sh $R/runs/A_gptoss_128k_run1/decode/server.log decode_A_final > $R/runs/A_gptoss_128k_run1/compile_audit_final.txt
bash $S/pd.sh stop; sleep 5; bash $S/tpu_free.sh || { log "TPU not free, abort"; exit 1; }
RUN=A_qwen_128k_run1; log "starting PD Config A qwen3-vl 128k -> $RUN"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/qwen3-vl_A_128k.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log
nohup $PY $S/pd_watchdog.py --server prefill=$PRE --server decode=$DEC --log decode=$R/runs/$RUN/decode/server.log --interval 10 --hang-secs 600 --out $R/runs/$RUN/watchdog.csv > $R/runs/$RUN/watchdog.out 2>&1 &
mkdir -p $R/A/tc1/$M $R/A/tiny/$M $R/A/mm/$M $R/A/chat/$M $R/A/abort/$M
log "(1) TC-1 c=1 (eager decoder: ~3 min per 1,024-token output)"
bash $S/run_tc1.sh $M $LB $R/A/tc1/$M 1024,8192,32768,131072,127999 $R/golden/qwen3-vl_fp8 1
for L in 1024 8192 32768 131072 127999; do python3 $S/needle_check.py --prompts $P/$L.jsonl --out $R/A/tc1/$M/$L.jsonl --other $R/golden/qwen3-vl_fp8/$L.jsonl --no-fail --report $R/A/tc1/$M/needles_$L.json 2>/dev/null | cut -c1-200; done
log "first-token logprob PD vs golden(c2) per prompt:"
$PY - $R/A/tc1/$M $R/golden/qwen3-vl_fp8 <<'EOF'
import json,sys,os
for L in [1024,8192,32768,131072,127999]:
    a={json.loads(l)['id']:json.loads(l) for l in open(f"{sys.argv[2]}/{L}.jsonl")}; b={json.loads(l)['id']:json.loads(l) for l in open(f"{sys.argv[1]}/{L}.jsonl")}
    for k in a:
        la=a[k].get('first_token_logprob'); lb=b.get(k,{}).get('first_token_logprob')
        print(f"   {k:14s} golden={la} pd={lb} delta={None if la is None or lb is None else round(abs(la-lb),4)} pd_prompt_tokens={b.get(k,{}).get('prompt_tokens')} golden_prompt_tokens={a[k].get('prompt_tokens')}")
EOF
state
log "(2) tiny boundary sweep (page 128, no SWA) c=1"
$PY $S/golden_diff.py generate --prompts $P/tiny.jsonl --url $LB --out $R/A/tiny/$M/tiny.jsonl --max-new-tokens 160 --ignore-eos --return-logprob --concurrency 1 --timeout 900 > $R/A/tiny/$M/generate.log 2>&1
grep -E 'errors' $R/A/tiny/$M/generate.log
log "(3) multimodal matrix via chat endpoint"
for spec in "1 448x448" "4 448x448" "16 448x448" "1 1024x1024" "1 1920x1080" "4 1024x1024"; do set -- $spec
  $PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images $1 --resolution $2 --max-tokens 192 --temperature 0 --needle-word plum-7731 --out $R/A/mm/$M/img${1}_${2}.json --timeout 900 2>&1 | grep -E 'prompt_tokens|usage|TTFT|needle|FOUND|MISSING|error|Error|HTTP' | sed "s/^/   n=$1 $2: /" | head -4
done
log "(3b) image straddling the 8k chunk, 64 images, text-only, corrupt image"
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 2 --resolution 448x448 --target-tokens 20000 --image-straddle-chunk 8150 --max-tokens 128 --temperature 0 --needle-word kiwi-4410 --out $R/A/mm/$M/straddle.json --timeout 900 2>&1 | grep -E 'prompt_tokens|usage|needle|FOUND|MISSING|error|HTTP' | sed 's/^/   straddle: /' | head -3
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 64 --resolution 448x448 --max-tokens 64 --temperature 0 --out $R/A/mm/$M/img64.json --timeout 1800 2>&1 | grep -E 'payload|prompt_tokens|usage|error|HTTP|TTFT' | sed 's/^/   64img: /' | head -4
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 0 --max-tokens 64 --temperature 0 --question "Reply with the word hello." --out $R/A/mm/$M/text_only.json --timeout 300 2>&1 | grep -E 'usage|error|HTTP|hello' | sed 's/^/   text-only: /' | head -2
$PY $S/chat_suite.py --model qwen3-vl-30b --kind qwen --a $LB --report $R/A/chat/$M/chat_pd_A.json --timeout 900 > $R/A/chat/$M/chat_pd_A.txt 2>&1; log "chat rc=$?"
state
log "(4) same image + different question (cache key must include image), then different image + same text"
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 1 --resolution 448x448 --max-tokens 64 --temperature 0 --question "What color is the shape?" --out $R/A/mm/$M/cache_q1.json 2>&1 | grep -E 'usage|cached' | sed 's/^/   q1: /'
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 1 --resolution 448x448 --max-tokens 64 --temperature 0 --question "What number is written in the image?" --out $R/A/mm/$M/cache_q2.json 2>&1 | grep -E 'usage|cached' | sed 's/^/   q2: /'
log "(5) aborts (decode phase; eager decode makes this long)"
$PY $S/abort_test.py --url $LB --prompt-file $P/32768.jsonl --phase decode --abort-after 10 --repeat 2 --server-info $PRE --server-info $DEC --settle 90 --report $R/A/abort/$M/abort_decode.json > $R/A/abort/$M/abort_decode.txt 2>&1; log "abort rc=$?"
sleep 60; state
bash $S/compile_audit.sh $R/runs/$RUN/decode/server.log decode_A_qwen | tee $R/A/compile_audit_decode_A_qwen.txt
grep -c 'VisionTpuCompilePolicy.run: compiled-fn' $R/runs/$RUN/decode/server.log | sed 's/^/   vision compile events (decode): /'
log "phase A qwen done"
