#!/usr/bin/env bash
# After the gpt-oss redo: redo the Qwen Config A items voided by the 1920x1080 crash. Crash-prone items last, with prefill relaunch.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; M=qwen3-vl; P=$HOME/pd_disagg/prompts/$M
log(){ echo "$(date -u +%FT%TZ) $*"; }
state(){ for u in $PRE $DEC; do echo "   $u $(curl -s -m 10 $u/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|full_token_usage|num_used_tokens)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' ')"; done; }
relaunch_prefill(){ ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217 "sudo -n -iu alisachen_google_com bash -lc 'bash ~/pd_disagg/scripts/cleanup.sh >/dev/null 2>&1; mv ~/pd_disagg/results/runs/$1/prefill/server.log ~/pd_disagg/results/runs/$1/prefill/server.\$(date -u +%H%M%S).log; cd /tmp && exec >/dev/null 2>&1 </dev/null; setsid nohup bash ~/pd_disagg/scripts/launch_role.sh ~/pd_disagg/recipes/qwen3-vl_A_128k.json prefill $1 >> ~/pd_disagg/results/runs/$1/launcher_prefill.out 2>&1 </dev/null &'"; until curl -sf -m 5 $PRE/health >/dev/null && curl -sf -m 30 $LB/health_generate >/dev/null; do sleep 20; done; log "prefill + LB back"; }
until grep -q 'phase A gpt-oss redo done' $R/phase_A_gptoss_redo.log 2>/dev/null; do sleep 60; done; log "gpt-oss redo done"
bash $S/pd.sh stop; sleep 5; bash $S/tpu_free.sh || { log "TPU not free, abort"; exit 1; }
RUN=A_qwen_128k_run2; log "PD Config A qwen 128k -> $RUN"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/qwen3-vl_A_128k.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log
nohup $PY $S/pd_watchdog.py --server prefill=$PRE --server decode=$DEC --log decode=$R/runs/$RUN/decode/server.log --interval 10 --hang-secs 600 --out $R/runs/$RUN/watchdog.csv > $R/runs/$RUN/watchdog.out 2>&1 &
MM=$R/A/mm/$M; mkdir -p $MM $R/A/chat/$M $R/A/abort/$M
log "(1) multimodal: straddle, 64 images, text-only, image-only, cache key checks, 4x1024"
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 2 --resolution 448x448 --target-tokens 20000 --image-straddle-chunk 8150 --max-tokens 128 --temperature 0 --needle-word kiwi-4410 --out $MM/straddle.json --timeout 900 2>&1 | grep -E 'status=|needle' | sed 's/^/   straddle: /'
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 64 --resolution 448x448 --max-tokens 64 --temperature 0 --out $MM/img64.json --timeout 1800 2>&1 | grep -E 'payload|status=' | sed 's/^/   64img: /'
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 0 --max-tokens 64 --temperature 0 --question "Reply with the word hello." --out $MM/text_only.json --timeout 300 2>&1 | grep -E 'status=' | sed 's/^/   text-only: /'
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 1 --resolution 448x448 --max-tokens 64 --temperature 0 --question "" --out $MM/image_only.json --timeout 300 2>&1 | grep -E 'status=' | sed 's/^/   image-only: /'
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 1 --resolution 448x448 --max-tokens 64 --temperature 0 --question "What color is the shape?" --out $MM/cache_q1.json 2>&1 | grep -E 'status=' | sed 's/^/   q1: /'
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 1 --resolution 448x448 --max-tokens 64 --temperature 0 --question "What number is written in the image?" --out $MM/cache_q2.json 2>&1 | grep -E 'status=' | sed 's/^/   q2 (same image, new question): /'
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 4 --resolution 1024x1024 --max-tokens 96 --temperature 0 --needle-word plum-7731 --out $MM/img4_1024x1024.json --timeout 900 2>&1 | grep -E 'status=|needle' | sed 's/^/   4x1024: /'
state
log "(2) chat suite (qwen kind)"; $PY $S/chat_suite.py --model qwen3-vl-30b --kind qwen --a $LB --report $R/A/chat/$M/chat_pd_A.json --timeout 900 > $R/A/chat/$M/chat_pd_A.txt 2>&1; log "chat rc=$?"; tail -n 12 $R/A/chat/$M/chat_pd_A.txt | cut -c1-170
log "(3) decode-phase aborts"; $PY $S/abort_test.py --url $LB --prompt-file $P/32768.jsonl --phase decode --abort-after 10 --repeat 2 --server-info $PRE --server-info $DEC --settle 90 --report $R/A/abort/$M/abort_decode.json > $R/A/abort/$M/abort_decode.txt 2>&1; log "abort rc=$?"; tail -n 5 $R/A/abort/$M/abort_decode.txt | cut -c1-170
sleep 15; curl -sf -m 5 $PRE/health >/dev/null || relaunch_prefill $RUN
log "(4) F-10 repro LAST: new vision resolution 1920x1080 (expect prefill crash), then 896x504 (another non-square), relaunch between"
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 1 --resolution 1920x1080 --max-tokens 48 --temperature 0 --out $MM/img1_1920x1080_repro.json --timeout 600 2>&1 | grep -E 'status=|est' | sed 's/^/   1920x1080: /'
sleep 20; curl -s -m 5 -o /dev/null -w "   prefill /health after 1920x1080: %{http_code}\n" $PRE/health; curl -sf -m 5 $PRE/health >/dev/null || relaunch_prefill $RUN
$PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 1 --resolution 896x504 --max-tokens 48 --temperature 0 --out $MM/img1_896x504.json --timeout 600 2>&1 | grep -E 'status=|est' | sed 's/^/   896x504: /'
sleep 20; curl -s -m 5 -o /dev/null -w "   prefill /health after 896x504: %{http_code}\n" $PRE/health; state
log "phase A qwen redo done"
