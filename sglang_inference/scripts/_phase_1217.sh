#!/usr/bin/env bash
# PR #1217 (teammate fix for #1147): n>1 through the pair on its head, boundary cases, main control, monolithic non-PD check.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; OUT=$R/fix_validate_1217; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [1217] $*"; }
posts(){ echo "prefill POSTs $($SSH_A "sudo -n grep -cE 'POST /generate HTTP/1.1\" 200' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log") decode POSTs $(grep -cE 'POST /generate HTTP/1.1" 200' $R/runs/$1/decode/server.log)"; }
used(){ curl -s -m 5 $DEC/metrics | grep -E '^sglang:num_used_tokens\{[^}]*tp_rank="0"' | head -n1 | sed -E 's/.* //'; }
excs(){ echo "   scheduler exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception|AssertionError' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception|AssertionError' $R/runs/$1/decode/server.log); health prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health); decode used tokens $(used)"; }
gen(){ # $1 name $2 json $3 timeout
  t0=$(date +%s.%N); code=$(curl -s -m $3 $LB/generate -H 'Content-Type: application/json' -d "$2" -o $OUT/$1.json -w '%{http_code}'); t1=$(date +%s.%N)
  n=$(python3 -c "import json; d=json.load(open('$OUT/$1.json')); print(len(d) if isinstance(d,list) else 1)" 2>/dev/null); echo "   $1: HTTP ${code:-000} outputs=${n:-?} in $(python3 -c "print(round($t1-$t0,1))")s $(python3 -c "import json; d=json.load(open('$OUT/$1.json')); d=d if isinstance(d,list) else [d]; print('tokens', [x.get('meta_info',{}).get('completion_tokens') for x in d])" 2>/dev/null)"
}
deploy(){ cd $HOME/sglang-torchtpu && git checkout -q main && git fetch -q origin amandaliang-sglang && git bundle create /tmp/p1217.bundle origin/amandaliang-sglang 2>/dev/null
  scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/p1217.bundle sa_101411256518727260534@10.128.15.217:/tmp/p1217.bundle
  $SSH_A "sudo -n chown alisachen_google_com /tmp/p1217.bundle; $GA fetch -q /tmp/p1217.bundle +origin/amandaliang-sglang:pr1217 && $GA checkout -q --detach pr1217 && echo '   host a on' \$($GA log -1 --format=%h)"
  git checkout -q --detach origin/amandaliang-sglang && log "local on $(git log -1 --format=%h)"; }
start(){ bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh start $1 $2 2>&1 | grep -E 'ready|NOT'; curl -sf -m 30 $LB/health_generate >/dev/null; }

deploy
log "=== 1217/A n>1 through the pair on the PR head (Config A)"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k.json v1217_pair; then
  gen n2 '{"text":"Name a fruit.","sampling_params":{"temperature":0.8,"max_new_tokens":32,"n":2}}' 300; echo "      $(posts v1217_pair) (expect prefill = decode = 2: no prefix request in PD)"
  gen n3 '{"text":"Name a fruit.","sampling_params":{"temperature":0.8,"max_new_tokens":32,"n":3}}' 300
  gen n8 '{"text":"Name a fruit.","sampling_params":{"temperature":0.8,"max_new_tokens":16,"n":8}}' 300
  gen batch2_n2 '{"text":["Name a fruit.","Name a color."],"sampling_params":{"temperature":0.8,"max_new_tokens":16,"n":2}}' 300
  gen batch3_n3 '{"text":["Name a fruit.","Name a color.","Name a city."],"sampling_params":{"temperature":0.8,"max_new_tokens":16,"n":3}}' 300
  for i in 1 2 3 4 5; do gen repeat${i}_n2 '{"text":"Say one word.","sampling_params":{"temperature":0.8,"max_new_tokens":8,"n":2}}' 120; done
  code=$(curl -s -m 300 $LB/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Name a fruit."}],"n":2,"max_tokens":16}' -o $OUT/chat_n2.json -w '%{http_code}'); echo "   chat n=2: HTTP $code choices=$(python3 -c "import json; print(len(json.load(open('$OUT/chat_n2.json')).get('choices',[])))" 2>/dev/null)"
  gen n1 '{"text":"Name a fruit.","sampling_params":{"temperature":0,"max_new_tokens":16,"n":1}}' 120
  sleep 5; excs v1217_pair; echo "      totals: $(posts v1217_pair)"
else log "1217/A pair did not come up"; fi
log "=== 1217/B control on main: n=2 through the pair (before: no response until the client gives up)"
bash $S/pd.sh stop >/dev/null 2>&1; $SSH_A "$GA checkout -q main"; cd $HOME/sglang-torchtpu && git checkout -q main
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k.json v1217_main; then
  gen main_n2 '{"text":"Name a fruit.","sampling_params":{"temperature":0.8,"max_new_tokens":32,"n":2}}' 90; sleep 3; excs v1217_main; echo "      $(posts v1217_main)"
else log "1217/B pair did not come up"; fi
log "=== 1217/C monolithic on the PR head: non-PD path keeps working (n=2, n=3 with the prefix request)"
bash $S/pd.sh stop >/dev/null 2>&1; cd $HOME/sglang-torchtpu && git checkout -q --detach origin/amandaliang-sglang
sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_fp8_e4m3.json v1217_mono b 2>&1 | grep -E 'ready|NOT'
LB=$DEC; gen mono_n2 '{"text":"Name a fruit.","sampling_params":{"temperature":0.8,"max_new_tokens":32,"n":2}}' 300; gen mono_n3 '{"text":"Name a fruit.","sampling_params":{"temperature":0.8,"max_new_tokens":32,"n":3}}' 300
echo "   mono POSTs: $(grep -cE 'POST /generate HTTP/1.1" 200' $R/runs/v1217_mono/golden/server.log) (2 requests); exceptions $(grep -cE 'Scheduler hit an exception' $R/runs/v1217_mono/golden/server.log); health $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
bash $S/pd.sh stop >/dev/null 2>&1; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase 1217 done"
