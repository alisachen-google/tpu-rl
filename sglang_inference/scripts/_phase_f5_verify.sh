#!/usr/bin/env bash
# #1244 (F-5): prompt-logprob request must be refused with 400 at intake, not kill the engine. Monolithic gpt-oss on host b.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; OUT=$R/fix_validate_1244; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
URL=http://10.128.0.90:30000; REC=$HOME/pd_disagg/recipes/gpt-oss_G_128k_fp8_e4m3.json; REC_ON=$HOME/pd_disagg/recipes/gpt-oss_G_128k_fp8_e4m3_promptlogprob.json
BODY='{"text":"The three primary colors are","return_logprob":true,"logprob_start_len":0,"sampling_params":{"temperature":0,"max_new_tokens":8}}'
PLAIN='{"text":"The three primary colors are","sampling_params":{"temperature":0,"max_new_tokens":8}}'
OUTONLY='{"text":"The three primary colors are","return_logprob":true,"sampling_params":{"temperature":0,"max_new_tokens":8}}'
log(){ echo "$(date -u +%FT%TZ) [f5] $*"; }
req(){ curl -s -m 120 -o $OUT/$1.json -w '%{http_code}' $URL/generate -H 'Content-Type: application/json' -d "$2"; }
health(){ curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health; }
run_case(){ # $1 ref  $2 tag  $3 recipe
  cd $HOME/sglang-torchtpu && git checkout -q --detach $1 && log "=== $2: $(git log -1 --format='%h %s' | cut -c1-80)"
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh golden $3 G_f5_$2 b 2>&1 | grep -E 'ready|NOT'
  L=$R/runs/G_f5_$2/golden/server.log
  echo "   output-only logprobs: HTTP $(req ${2}_outonly "$OUTONLY") $(python3 -c "import json; d=json.load(open('$OUT/${2}_outonly.json')); m=d.get('meta_info',{}); print('output_token_logprobs', len(m.get('output_token_logprobs') or []))" 2>/dev/null); health $(health)"
  code=$(req ${2}_prompt "$BODY"); echo "   prompt logprobs (logprob_start_len=0): HTTP $code; body: $(head -c 220 $OUT/${2}_prompt.json | tr '\n' ' ')"
  python3 -c "import json; d=json.load(open('$OUT/${2}_prompt.json')); m=d.get('meta_info',{}); print('   input_token_logprobs returned:', len(m.get('input_token_logprobs') or []))" 2>/dev/null
  sleep 5; echo "   health after: $(health); follow-up plain request: HTTP $(req ${2}_followup "$PLAIN"); scheduler exceptions in log: $(grep -cE 'Scheduler hit an exception' $L); NotImplementedError: $(grep -cE 'NotImplementedError: input/prompt' $L)"
  bash $S/pd.sh stop >/dev/null 2>&1
}
run_case tpu-prompt-logprob-validate fix_flag_off $REC
run_case tpu-prompt-logprob-validate fix_flag_on $REC_ON
run_case origin/main main_flag_off $REC
cd $HOME/sglang-torchtpu && git checkout -q main && log "phase f5 verify done"
