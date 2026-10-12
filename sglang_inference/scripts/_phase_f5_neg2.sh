#!/bin/bash
# #1245: logprob_start_len=-2 (unvalidated upstream, clamps to 0) must be refused with 400, flag off.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; OUT=$R/fix_validate_1244_v2; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
URL=http://10.128.0.90:30000; REC=$HOME/pd_disagg/recipes/gpt-oss_G_128k_fp8_e4m3.json
log(){ echo "$(date -u +%FT%TZ) [f5neg] $*"; }
req(){ curl -s -m 120 -o $OUT/$1.json -w '%{http_code}' $URL/generate -H 'Content-Type: application/json' -d "$2"; }
body(){ echo "{\"text\":\"The three primary colors are\",\"return_logprob\":true,\"logprob_start_len\":$1,\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":8}}"; }
cd $HOME/sglang-torchtpu && git fetch -q origin && git checkout -q --detach origin/tpu-prompt-logprob-validate && log "=== neg2 on $(git log -1 --format=%h)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh golden $REC G_f5neg b 2>&1 | grep -E 'ready|NOT'
for s in -2 -7 -1 0; do code=$(req neg_$s "$(body $s)"); echo "   logprob_start_len=$s: HTTP $code; $(head -c 160 $OUT/neg_$s.json | tr '\n' ' ')"; done
sleep 3; echo "   health after: $(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health); follow-up plain: HTTP $(req neg_followup '{"text":"Hello","sampling_params":{"max_new_tokens":4}}')"
bash $S/pd.sh stop >/dev/null 2>&1; git checkout -q main; log "phase f5neg done"
