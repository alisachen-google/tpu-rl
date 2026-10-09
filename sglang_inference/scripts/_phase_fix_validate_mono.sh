#!/usr/bin/env bash
# F-11 trigger 1 end-to-end without F-12: monolithic gpt-oss at the 1M shape, radix off, 1 slot, on host b.
# (a) fix branch -> 256k prompt must return tokens; (b) main -> expected to stall (F-11), capped at 900 s.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
URL=http://10.128.0.90:30000; P=$HOME/pd_disagg/prompts/gpt-oss; OUT=$R/fix_validate; mkdir -p $OUT; REC=$HOME/pd_disagg/recipes/gpt-oss_G_1m_mr1_noradix.json
log(){ echo "$(date -u +%FT%TZ) [fixval-mono] $*"; }
run_case(){ # $1 branch  $2 run-name  $3 client timeout
  cd $HOME/sglang-torchtpu && git checkout -q $1 && log "checked out $(git branch --show-current) ($(git log -1 --format=%h))"
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $REC $2 b 2>&1 | grep -E 'ready|NOT'
  echo "   SWA pool tokens: $(grep -E 'SWA KV Cache is allocated' $R/runs/$2/*/server.log $R/runs/$2/server.log 2>/dev/null | head -n 1 | sed -E 's/.*#tokens: ([0-9]+).*/\1/')"
  t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $P/262144.jsonl --url $URL --out $OUT/mono_$2.jsonl --max-new-tokens 32 --ignore-eos --concurrency 1 --timeout $3 --stream > $OUT/mono_$2.log 2>&1
  echo "   262144: $(grep -E 'tokens=|ERROR' $OUT/mono_$2.log | head -n 1 | cut -c1-150) ($(( $(date +%s)-t0 ))s)"
  L=$(ls $R/runs/$2/*/server.log $R/runs/$2/server.log 2>/dev/null | head -n 1); echo "   chunk trail: $(grep -E 'Prefill batch' $L | sed -E 's/.*#new-token: ([0-9]+).*swa token usage: ([0-9.]+).*#pending-token: ([0-9]+).*/\1@\2/' | tail -n 6 | tr '\n' ' ')  batches=$(grep -cE 'Prefill batch' $L)"
  echo "   health: $(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health)"
  bash $S/pd.sh stop >/dev/null 2>&1
}
log "=== (a) fix branch pd-fixes-validate"; run_case pd-fixes-validate G_1m_mr1_noradix_fix 1200
log "=== (b) main (expected F-11 stall)"; run_case main G_1m_mr1_noradix_main 900
cd $HOME/sglang-torchtpu && git checkout -q main; log "phase fix validate mono done (checkout: $(git branch --show-current))"
