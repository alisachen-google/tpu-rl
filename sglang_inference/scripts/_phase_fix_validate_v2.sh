#!/usr/bin/env bash
# Live validation of PRs #1191 (#1188 queue limit), #1192 (#1147 n>1 rooms), #1193 (#1176 canary) on the pair.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; P=$HOME/pd_disagg/prompts/gpt-oss; OUT=$R/fix_validate_v2; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [fixv2] $*"; }
health(){ echo "   health: prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health); used: prefill $(curl -s -m 5 $PRE/metrics | grep -E '^sglang:num_used_tokens' | head -n1 | sed -E 's/.* //') decode $(curl -s -m 5 $DEC/metrics | grep -E '^sglang:num_used_tokens' | head -n1 | sed -E 's/.* //')"; }
excs(){ echo "   exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception' $R/runs/$1/decode/server.log)"; }
start(){ bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $1 $2 2>&1 | tee $R/runs/$2.driver.log | grep -E 'ready|NOT'; curl -sf -m 30 $LB/health_generate >/dev/null; }
cd $HOME/sglang-torchtpu && git bundle create /tmp/fixv2.bundle pd-fixes-validate2 2>/dev/null && log "validate head: $(git log -1 --format=%h pd-fixes-validate2)"
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/fixv2.bundle sa_101411256518727260534@10.128.15.217:/tmp/fixv2.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/fixv2.bundle; $GA fetch -q /tmp/fixv2.bundle pd-fixes-validate2:pd-fixes-validate2 && $GA checkout -q pd-fixes-validate2 && echo '   host a on' \$($GA log -1 --format=%h)"
git checkout -q pd-fixes-validate2 && log "local on $(git log -1 --format=%h)"
rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' $HOME/pd_disagg/recipes/ sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/recipes/

log "=== V1 #1188 queue limit on the pair: Config C + --max-queued-requests 4 + priority scheduling"
if start $HOME/pd_disagg/recipes/gpt-oss_C_128k_ps_q.json fixv2_C_ps_q; then
  cd $S && $PY priority_probe.py --url $LB --decode $DEC --mode q --slots 8 --hold-tokens 3000 > $OUT/v1_ps_q.txt 2>&1; grep -E '^\[D|^\[end' $OUT/v1_ps_q.txt | sed 's/^/   /' | cut -c1-230; excs fixv2_C_ps_q; health
else log "V1 pair did not come up"; fi

log "=== V2 #1147 n>1 through the pair: Config A"
if start $HOME/pd_disagg/recipes/gpt-oss_A_128k.json fixv2_A_n; then
  for n in 2 3; do
    t0=$(date +%s); curl -s -m 300 $LB/generate -H 'Content-Type: application/json' -d "{\"text\":\"List three colours.\",\"sampling_params\":{\"n\":$n,\"temperature\":0.7,\"max_new_tokens\":32}}" -o $OUT/v2_n$n.json -w "   n=$n: HTTP %{http_code} in %{time_total}s\n"
    python3 -c "
import json; d=json.load(open('$OUT/v2_n$n.json')); d=d if isinstance(d,list) else [d]; print('   outputs:', len(d), 'completion_tokens:', [ (x.get('meta_info') or {}).get('completion_tokens') for x in d], 'rooms distinct on client side: n/a')"
  done
  curl -s -m 300 $LB/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Name a fruit."}],"n":2,"max_tokens":16}' -o $OUT/v2_chat_n2.json -w "   chat n=2: HTTP %{http_code} in %{time_total}s\n"; python3 -c "
import json; d=json.load(open('$OUT/v2_chat_n2.json')); print('   choices:', len(d.get('choices',[])), [c.get('finish_reason') for c in d.get('choices',[])])"
  sleep 10; excs fixv2_A_n; health
  echo "   decode rooms seen for n-requests (distinct bootstrap_room values in decode log tail): $(grep -oE 'bootstrap_room=[0-9]+' $R/runs/fixv2_A_n/decode/server.log | sort -u | wc -l)"
else log "V2 pair did not come up"; fi

log "=== V3 #1176 --kv-canary log: gpt-oss Config A (SWA pool) then Qwen3-VL Config A (fused pool)"
for m in gpt-oss qwen3-vl; do
  if start $HOME/pd_disagg/recipes/${m}_A_128k_canary.json fixv2_${m}_canary; then
    PP=$HOME/pd_disagg/prompts/$m
    $PY $S/golden_diff.py generate --prompts $PP/8192.jsonl --url $LB --out $OUT/v3_${m}_8k.jsonl --max-new-tokens 64 --ignore-eos --concurrency 2 --timeout 900 --stream > $OUT/v3_${m}_8k.log 2>&1; echo "   $m 8k: completed $(grep -cE 'tokens=' $OUT/v3_${m}_8k.log)/2 errors $(grep -cE 'ERROR' $OUT/v3_${m}_8k.log)"
    $PY $S/golden_diff.py generate --prompts $PP/32768.jsonl --url $LB --out $OUT/v3_${m}_32k.jsonl --max-new-tokens 64 --ignore-eos --concurrency 2 --timeout 900 --stream > $OUT/v3_${m}_32k.log 2>&1; echo "   $m 32k: completed $(grep -cE 'tokens=' $OUT/v3_${m}_32k.log)/2 errors $(grep -cE 'ERROR' $OUT/v3_${m}_32k.log)"
    echo "   canary lines: prefill $($SSH_A "sudo -n grep -ciE 'canary' /home/alisachen_google_com/pd_disagg/results/runs/fixv2_${m}_canary/prefill/server.log") decode $(grep -ciE 'canary' $R/runs/fixv2_${m}_canary/decode/server.log); violations: prefill $($SSH_A "sudo -n grep -ciE 'canary.*(violation|mismatch|corrupt)' /home/alisachen_google_com/pd_disagg/results/runs/fixv2_${m}_canary/prefill/server.log") decode $(grep -ciE 'canary.*(violation|mismatch|corrupt)' $R/runs/fixv2_${m}_canary/decode/server.log)"
    grep -iE 'canary' $R/runs/fixv2_${m}_canary/decode/server.log | grep -vE 'server_args|UserWarning' | head -n 3 | cut -c1-200 | sed 's/^/   decode: /'
    excs fixv2_${m}_canary; health
  else log "V3 $m pair did not come up"; $SSH_A "sudo -n grep -E 'Error|error' /home/alisachen_google_com/pd_disagg/results/runs/fixv2_${m}_canary/prefill/server.log | grep -vE UserWarning | tail -n 3 | cut -c1-220" | sed 's/^/   prefill: /'; grep -E 'Error|error' $R/runs/fixv2_${m}_canary/decode/server.log | grep -vE UserWarning | tail -n 3 | cut -c1-220 | sed 's/^/   decode: /'; fi
done
bash $S/pd.sh stop >/dev/null 2>&1
$SSH_A "$GA checkout -q main && echo '   host a restored to' \$($GA log -1 --format=%h)"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase fix validate v2 done (local on $(git branch --show-current))"
