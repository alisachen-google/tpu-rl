#!/bin/bash
# Re-validate #1245 after the redesign (scheduler-intake patch): fix with flag off / on. main baseline measured earlier.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; OUT=$R/fix_validate_1244_v2; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
URL=http://10.128.0.90:30000; REC=$HOME/pd_disagg/recipes/gpt-oss_G_128k_fp8_e4m3.json; REC_ON=$HOME/pd_disagg/recipes/gpt-oss_G_128k_fp8_e4m3_promptlogprob.json
BODY='{"text":"The three primary colors are","return_logprob":true,"logprob_start_len":0,"sampling_params":{"temperature":0,"max_new_tokens":8}}'
BODY2='{"text":"The three primary colors are","return_logprob":true,"logprob_start_len":2,"sampling_params":{"temperature":0,"max_new_tokens":8}}'
PLAIN='{"text":"The three primary colors are","sampling_params":{"temperature":0,"max_new_tokens":8}}'
OUTONLY='{"text":"The three primary colors are","return_logprob":true,"sampling_params":{"temperature":0,"max_new_tokens":8}}'
log(){ echo "$(date -u +%FT%TZ) [f5v2] $*"; }
req(){ curl -s -m 120 -o $OUT/$1.json -w '%{http_code}' $URL/generate -H 'Content-Type: application/json' -d "$2"; }
health(){ curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health; }
run_case(){ # $1 ref  $2 tag  $3 recipe
  cd $HOME/sglang-torchtpu && git fetch -q origin && git checkout -q --detach $1 && log "=== $2: $(git log -1 --format='%h %s' | cut -c1-80)"
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh golden $3 G_f5v2_$2 b 2>&1 | grep -E 'ready|NOT'
  L=$R/runs/G_f5v2_$2/golden/server.log
  echo "   plain: HTTP $(req ${2}_plain "$PLAIN")"
  echo "   output-only logprobs: HTTP $(req ${2}_outonly "$OUTONLY") $(python3 -c "import json; d=json.load(open('$OUT/${2}_outonly.json')); m=d.get('meta_info',{}); print('output_token_logprobs:', len(m.get('output_token_logprobs') or []), 'input:', len(m.get('input_token_logprobs') or []))" 2>&1)"
  for b in BODY BODY2; do code=$(req ${2}_prompt_$b "${!b}"); echo "   prompt logprobs ($b): HTTP $code; body: $(head -c 260 $OUT/${2}_prompt_$b.json | tr '\n' ' ')"; done
  python3 -c "import json; d=json.load(open('$OUT/${2}_prompt_BODY.json')); m=d.get('meta_info',{}); print('   input_token_logprobs returned:', len(m.get('input_token_logprobs') or []))" 2>/dev/null
  sleep 5; echo "   health after: $(health); follow-up plain: HTTP $(req ${2}_followup "$PLAIN"); burst 8 plain: $(for i in $(seq 8); do req ${2}_burst_$i "$PLAIN"; echo -n ' '; done); scheduler tracebacks: $(grep -cE 'Traceback|NotImplementedError' $L 2>/dev/null)"
  bash $S/pd.sh stop >/dev/null 2>&1
}
run_case origin/tpu-prompt-logprob-validate fix_flag_off $REC
run_case origin/tpu-prompt-logprob-validate fix_flag_on $REC_ON
cd $HOME/sglang-torchtpu && git checkout -q main && log "phase f5v2 done"
