#!/usr/bin/env bash
# After the goldens-1m/F4 phase: redo the Config A gpt-oss items that were voided by F-8 (chat suite, GSM8K via LB, F-6 repro,
# F-5 repro, aborts). Crash-prone repros run last; after each engine death the prefill is relaunched.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; M=gpt-oss; P=$HOME/pd_disagg/prompts/$M
log(){ echo "$(date -u +%FT%TZ) $*"; }
state(){ for u in $PRE $DEC; do echo "   $u $(curl -s -m 10 $u/metrics | grep -E '^sglang:(num_running_reqs|num_queue_reqs|full_token_usage|num_used_tokens)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' ')"; done; }
relaunch_prefill(){ # relaunch rank 0 (prefill + LB) on host a for RUN=$1
  ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217 "sudo -n -iu alisachen_google_com bash -lc 'bash ~/pd_disagg/scripts/cleanup.sh >/dev/null 2>&1; mv ~/pd_disagg/results/runs/$1/prefill/server.log ~/pd_disagg/results/runs/$1/prefill/server.\$(date -u +%H%M%S).log; cd /tmp && exec >/dev/null 2>&1 </dev/null; setsid nohup bash ~/pd_disagg/scripts/launch_role.sh ~/pd_disagg/recipes/gpt-oss_A_128k.json prefill $1 >> ~/pd_disagg/results/runs/$1/launcher_prefill.out 2>&1 </dev/null &'"
  until curl -sf -m 5 $PRE/health >/dev/null && curl -sf -m 30 $LB/health_generate >/dev/null; do sleep 20; done; log "prefill + LB back"
}
until grep -q 'phase goldens-1m + F4 done' $R/phase_goldens_1m_F4.log 2>/dev/null; do sleep 60; done; log "goldens/F4 phase done"
bash $S/pd.sh stop; sleep 5; bash $S/tpu_free.sh || { log "TPU not free, abort"; exit 1; }
RUN=A_gptoss_128k_run2; log "PD Config A gpt-oss 128k -> $RUN"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log
nohup $PY $S/pd_watchdog.py --server prefill=$PRE --server decode=$DEC --log decode=$R/runs/$RUN/decode/server.log --interval 10 --hang-secs 300 --out $R/runs/$RUN/watchdog.csv > $R/runs/$RUN/watchdog.out 2>&1 &
mkdir -p $R/A/chat/$M $R/A/abort/$M $R/A/params/$M
log "(1) chat suite"; $PY $S/chat_suite.py --model gpt-oss-120b --kind gptoss --a $LB --report $R/A/chat/$M/chat_pd_A.json --timeout 600 > $R/A/chat/$M/chat_pd_A.txt 2>&1; log "chat rc=$?"; tail -n 16 $R/A/chat/$M/chat_pd_A.txt | cut -c1-170
log "(2) GSM8K-100 via LB"; cd $HOME/sglang-torchtpu && $PY -m sglang.test.run_eval --base-url $LB --model gpt-oss-120b --eval-name gsm8k --num-examples 100 --num-threads 4 --api chat --max-tokens 2048 --temperature 0 > $R/A/gsm8k_pd_A.log 2>&1; grep -E 'gsm8k_score' $R/A/gsm8k_pd_A.log; state
log "(3) KV checksum is NOT on in this config; recording transfer stats"; curl -s -m 10 $DEC/metrics | grep -E '^sglang:kv_transfer_(latency_ms|total_mb|bootstrap_ms)_(sum|count)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' '; echo
log "(4) F-6 repro: n=2 greedy, then n=2 sampled, 120 s timeout each"
for t in 0 0.7; do curl -s -m 120 $LB/generate -H 'Content-Type: application/json' -d "{\"text\":\"The three primary colors are\",\"sampling_params\":{\"temperature\":$t,\"max_new_tokens\":32,\"n\":2}}" -o $R/A/params/$M/n2_repro_t$t.json -w "   n=2 t=$t HTTP %{http_code} in %{time_total}s\n"; head -c 300 $R/A/params/$M/n2_repro_t$t.json 2>/dev/null; echo; sleep 10; state; done
log "(5) F-5 repro: prompt logprobs (expect prefill crash) then relaunch"
curl -s -m 120 $LB/generate -H 'Content-Type: application/json' -d '{"text":"hi there","sampling_params":{"max_new_tokens":4},"return_logprob":true,"logprob_start_len":0}' -o $R/A/params/$M/f5_repro.json -w "   prompt-logprob HTTP %{http_code} in %{time_total}s\n"; head -c 300 $R/A/params/$M/f5_repro.json 2>/dev/null; echo
sleep 20; curl -s -m 5 -o /dev/null -w "   prefill /health after F-5 repro: %{http_code}\n" $PRE/health
curl -sf -m 5 $PRE/health >/dev/null || relaunch_prefill $RUN
log "(6) F-8 repro: single prefill-phase abort (expect crash), decode-phase aborts after relaunch"
$PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase prefill --abort-after 2 --repeat 1 --server-info $PRE --server-info $DEC --settle 30 --report $R/A/abort/$M/abort_prefill_repro.json > $R/A/abort/$M/abort_prefill_repro.txt 2>&1; log "abort prefill rc=$?"; tail -n 5 $R/A/abort/$M/abort_prefill_repro.txt | cut -c1-170
sleep 15; curl -s -m 5 -o /dev/null -w "   prefill /health after prefill-abort: %{http_code}\n" $PRE/health
curl -sf -m 5 $PRE/health >/dev/null || relaunch_prefill $RUN
$PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase decode --abort-after 5 --repeat 3 --concurrent 2 --server-info $PRE --server-info $DEC --settle 60 --report $R/A/abort/$M/abort_decode.json > $R/A/abort/$M/abort_decode.txt 2>&1; log "abort decode rc=$?"; tail -n 6 $R/A/abort/$M/abort_decode.txt | cut -c1-170
sleep 15; curl -s -m 5 -o /dev/null -w "   prefill /health after decode-aborts: %{http_code}\n" $PRE/health; state
log "phase A gpt-oss redo done"
