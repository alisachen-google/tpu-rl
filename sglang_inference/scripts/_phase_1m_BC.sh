#!/usr/bin/env bash
# 1M-context shape for Configs B and C (gpt-oss, radix on → ratio-sized SWA pool, so F-11 should not apply). TC-1 vs the 1m c=1 goldens at
# 32k, 256k, 512k, 1M and the unaligned lengths, needles, boundary 4xx, prefix reuse, GSM8K, decode aborts, compile audit.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; P=$HOME/pd_disagg/prompts/gpt-oss; G=$R/golden/gpt-oss_fp8_1m_c1
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
log(){ echo "$(date -u +%FT%TZ) [1m/$CFG] $*"; }
state(){ for u in $PRE $DEC; do echo "   $u $(curl -s -m 10 $u/metrics | grep -E '^sglang:(num_running_reqs|num_used_tokens|swa_token_usage)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' ')"; done; }
for CFG in B C; do
  OUT=$R/$CFG/gpt-oss_1m; mkdir -p $OUT; RUN=${CFG}_gpt-oss_1m; REC=$HOME/pd_disagg/recipes/gpt-oss_${CFG}_1m.json
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  log "=== start $REC"; t0=$(date +%s)
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $REC $RUN 2>&1 | tee $R/runs/$RUN.driver.log | grep -E 'ready|NOT'
  if ! curl -sf -m 30 $LB/health_generate >/dev/null; then log "PAIR DID NOT COME UP"; grep -hE 'Error|died|Check failed' $R/runs/$RUN/decode/server.log | grep -vE 'UserWarning|freeze_gc' | tail -n 3 | cut -c1-200; bash $S/pd.sh stop >/dev/null 2>&1; continue; fi
  log "pair up in $(( $(date +%s) - t0 )) s; SWA pool: $($SSH_A "sudo -n grep -m1 -oE 'aligning swa_max_total_num_tokens [0-9]+ -> [0-9]+' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log")"
  log "(0) probe: 32k prompt with 900 s timeout (F-11 bail-out)"
  $PY $S/golden_diff.py generate --prompts $P/32768.jsonl --url $LB --out $OUT/tc1_32768.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 --timeout 900 --stream > $OUT/tc1_generate_32768.log 2>&1
  if ! grep -q 'tokens=1024' $OUT/tc1_generate_32768.log; then log "!! 32k probe failed (F-11-class stall at 1M with radix on?)"; state; $SSH_A "sudo -n grep -E 'Prefill batch' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log | tail -n 3 | grep -oE '#new-token: [0-9]+.*swa token usage: [0-9.]+'" | sed 's/^/   /'; bash $S/pd.sh stop >/dev/null 2>&1; continue; fi
  $PY $S/golden_diff.py diff --a $G/32768.jsonl --b $OUT/tc1_32768.jsonl --report $OUT/tc1_diff_32768.json --expect-tokens 1024 2>&1 | grep -E '^isl|ids,' | sed 's/^/   vs golden /'
  log "(1) TC-1 long: 256k, 255,123, 512k, 511,777, 1M, 999,999 (c=1, 1024 tokens)"
  for L in 262144 255123 524288 511777 1047552 999999; do
    $PY $S/golden_diff.py generate --prompts $P/$L.jsonl --url $LB --out $OUT/tc1_$L.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 --timeout 3600 --stream > $OUT/tc1_generate_$L.log 2>&1
    grep -E 'tokens=|ERROR' $OUT/tc1_generate_$L.log | cut -c1-160 | sed 's/^/   /'
    [ -f $G/$L.jsonl ] && $PY $S/golden_diff.py diff --a $G/$L.jsonl --b $OUT/tc1_$L.jsonl --report $OUT/tc1_diff_$L.json --expect-tokens 1024 2>&1 | grep -E '^isl|ids,' | sed 's/^/   vs golden /'
    $PY $S/needle_check.py --prompts $P/$L.jsonl --out $OUT/tc1_$L.jsonl --no-fail --report $OUT/needles_$L.json 2>&1 | tail -n 2 | sed 's/^/   needles /'
    curl -sf -m 5 $PRE/health >/dev/null || { log "!! prefill down after $L"; break; }
  done
  state
  log "(2) boundary: ISL = context-1024 accepted, context+1 rejected"
  $PY $S/verify_pd_disagg.py --url $LB --router --lengths 1024 --boundary --context-length 1056768 --timeout 3600 2>&1 | tail -n 6 | sed 's/^/   /'
  log "(3) prefix reuse at 256k (second pass cached)"
  $PY $S/golden_diff.py generate --prompts $P/262144.jsonl --url $LB --out $OUT/reuse_262144.jsonl --max-new-tokens 64 --ignore-eos --concurrency 1 --timeout 1800 --stream > $OUT/reuse_262144.log 2>&1; grep -E 'cached=' $OUT/reuse_262144.log | cut -c1-140 | sed 's/^/   /'
  log "(4) GSM8K-100"
  cd $HOME/sglang-torchtpu && $PY -m sglang.test.run_eval --base-url $LB --model gpt-oss-120b --eval-name gsm8k --num-examples 100 --num-threads 4 --api chat --max-tokens 2048 --temperature 0 > $OUT/gsm8k.log 2>&1; grep -E 'gsm8k_score' $OUT/gsm8k.log | sed 's/^/   /'; cd $HOME
  log "(5) decode-phase aborts at 512k"
  $PY $S/abort_test.py --url $LB --prompt-file $P/524288.jsonl --phase decode --abort-after 20 --repeat 2 --server-info $PRE --server-info $DEC --settle 90 --report $OUT/abort_decode.json > $OUT/abort_decode.txt 2>&1; grep -E 'PASS|FAIL' $OUT/abort_decode.txt | tail -n 1 | cut -c1-200 | sed 's/^/   /'
  log "(6) compile audit"; bash $S/compile_audit.sh $R/runs/$RUN/decode/server.log decode_${CFG}_1m | head -n 3 | sed 's/^/   /'
  mkdir -p $R/logs/runs/$RUN; $SSH_A "sudo -n gzip -c /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log" > $R/logs/runs/$RUN/prefill_server.log.gz; gzip -c $R/runs/$RUN/decode/server.log > $R/logs/runs/$RUN/decode_server.log.gz
  bash $S/pd.sh stop >/dev/null 2>&1; log "=== $CFG at 1M done"
done
log "phase 1m BC done"
