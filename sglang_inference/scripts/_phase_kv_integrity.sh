#!/usr/bin/env bash
# TC-15: P/D KV-transfer integrity with the build's own instrumentation: --disaggregation-enable-kv-checksum (Adler-32 per request,
# verified on decode) and --kv-canary log (pool canary sweep every 50 steps, real-data hash 'partial') on both engines. Config A recipes.
# Runs TC-1 (8k/32k/128k, c=1) and decode-phase aborts, then counts checksum mismatches and canary reports in both logs.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
log(){ echo "$(date -u +%FT%TZ) [kv-integrity] $*"; }
until grep -q 'phase hicache reload done' $R/phase_hicache_reload.log 2>/dev/null; do sleep 120; done; log "hicache phase done"
for M in gpt-oss qwen3-vl; do
  OUT=$R/A_kvcheck/$M; mkdir -p $OUT; RUN=A_${M}_128k_kvcheck; P=$HOME/pd_disagg/prompts/$M
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  log "=== $M: start pair with kv checksum + canary"
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/${M}_A_128k_kvcheck.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log | grep -E 'ready|NOT'
  if ! curl -sf -m 30 $LB/health_generate >/dev/null; then log "pair did not come up"; grep -hE 'Error|died|Check failed|Traceback' $R/runs/$RUN/decode/server.log | grep -vE 'UserWarning|freeze_gc' | tail -n 4 | cut -c1-200; $SSH_A "sudo -n grep -hE 'Error|died|Check failed' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log | tail -n 4 | cut -c1-200"; bash $S/pd.sh stop >/dev/null 2>&1; continue; fi
  echo "   checksum enabled (prefill log): $($SSH_A "sudo -n grep -ciE 'checksum' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log")  canary lines at start (decode): $(grep -ciE 'canary' $R/runs/$RUN/decode/server.log)"
  log "(1) TC-1 8k/32k/128k c=1 with checksum verification on decode"
  for L in 8192 32768 131072; do
    $PY $S/golden_diff.py generate --prompts $P/$L.jsonl --url $LB --out $OUT/tc1_$L.jsonl --max-new-tokens 256 --ignore-eos --return-logprob --concurrency 1 --timeout 1800 --stream > $OUT/tc1_$L.log 2>&1
    grep -E 'tokens=|ERROR' $OUT/tc1_$L.log | cut -c1-160 | sed 's/^/   /'
    $PY $S/needle_check.py --prompts $P/$L.jsonl --out $OUT/tc1_$L.jsonl 2>&1 | tail -n 1 | sed 's/^/   /'
  done
  log "(2) concurrency 4 at 32k (checksum under load)"
  $PY $S/golden_diff.py generate --prompts $P/32768.jsonl --url $LB --out $OUT/c4_32k.jsonl --max-new-tokens 128 --ignore-eos --concurrency 4 --timeout 900 > $OUT/c4_32k.log 2>&1; grep -cE 'tokens=128' $OUT/c4_32k.log | sed 's/^/   completed: /'
  log "(3) decode-phase aborts (checksum + canary during abort/release)"
  $PY $S/abort_test.py --url $LB --prompt-file $P/131072.jsonl --phase decode --abort-after 5 --repeat 3 --concurrent 2 --server-info $PRE --server-info $DEC --settle 60 --report $OUT/abort_decode.json > $OUT/abort_decode.txt 2>&1; grep -E 'PASS|FAIL' $OUT/abort_decode.txt | tail -n 1 | cut -c1-160 | sed 's/^/   /'
  log "(4) integrity report"
  echo "   decode: checksum mismatches=$(grep -cE 'KV checksum mismatch' $R/runs/$RUN/decode/server.log)  canary reports=$(grep -ciE 'canary' $R/runs/$RUN/decode/server.log | sed 's/$/ (incl. startup)/')"
  grep -iE 'canary' $R/runs/$RUN/decode/server.log | grep -vE 'server_args|kv_canary' | tail -n 4 | cut -c1-200 | sed 's/^/   decode canary: /'
  $SSH_A "sudo -n bash -c \"echo '   prefill: checksum lines='\$(grep -ciE 'checksum' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log)' canary reports='\$(grep -ciE 'canary' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log); grep -iE 'canary|checksum' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log | grep -vE 'server_args|kv_canary|enable_kv_checksum' | tail -n 4 | cut -c1-200 | sed 's/^/   prefill: /'\""
  echo "   decode metrics: $(curl -s -m 10 $DEC/metrics | grep -iE 'checksum|canary' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' ')"
  mkdir -p $R/logs/runs/$RUN; $SSH_A "sudo -n gzip -c /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log" > $R/logs/runs/$RUN/prefill_server.log.gz; gzip -c $R/runs/$RUN/decode/server.log > $R/logs/runs/$RUN/decode_server.log.gz
  bash $S/pd.sh stop >/dev/null 2>&1
done
log "phase kv integrity done"
