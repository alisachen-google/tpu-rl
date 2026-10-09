#!/usr/bin/env bash
# #1176 follow-up: canary with torch.compile off (eager decode) on the PR head; does the canary serve and verify on TPU at all?
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/fix_validate_v2; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
GA="sudo -n -u alisachen_google_com git -C /home/alisachen_google_com/sglang-torchtpu -c safe.directory=*"
log(){ echo "$(date -u +%FT%TZ) [canary-eager] $*"; }
cd $HOME/sglang-torchtpu && git bundle create /tmp/canary.bundle tpu-kv-canary-fused-pool 2>/dev/null
scp -q -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/canary.bundle sa_101411256518727260534@10.128.15.217:/tmp/canary.bundle
$SSH_A "sudo -n chown alisachen_google_com /tmp/canary.bundle; $GA fetch -q /tmp/canary.bundle tpu-kv-canary-fused-pool:tpu-kv-canary-fused-pool && $GA checkout -q tpu-kv-canary-fused-pool && echo '   host a on' \$($GA log -1 --format=%h)"
git checkout -q --detach tpu-kv-canary-fused-pool && log "local on $(git log -1 --format=%h)"
rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' $HOME/pd_disagg/recipes/ sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/recipes/
for m in qwen3-vl gpt-oss; do
  log "=== $m Config A, --kv-canary log, enable_torch_compile=false"
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 2400 bash $S/pd.sh start $HOME/pd_disagg/recipes/${m}_A_128k_canary_eager.json canary_eager_$m 2>&1 | grep -E 'ready|NOT'
  if curl -sf -m 30 $LB/health_generate >/dev/null; then
    PP=$HOME/pd_disagg/prompts/$m
    $PY $S/golden_diff.py generate --prompts $PP/8192.jsonl --url $LB --out $OUT/canary_eager_${m}_8k.jsonl --max-new-tokens 32 --ignore-eos --concurrency 2 --timeout 900 --stream > $OUT/canary_eager_${m}_8k.log 2>&1; echo "   $m 8k: completed $(grep -cE 'tokens=' $OUT/canary_eager_${m}_8k.log)/2 errors $(grep -cE 'ERROR' $OUT/canary_eager_${m}_8k.log) $(grep -oE 'ttft=[0-9.]+ms' $OUT/canary_eager_${m}_8k.log | head -n 1)"
    for u in "prefill:$($SSH_A "sudo -n grep -iE 'canary' /home/alisachen_google_com/pd_disagg/results/runs/canary_eager_$m/prefill/server.log | grep -vE 'server_args|install_canary|attach_canary' | tail -n 3 | cut -c1-180" | tr '\n' '|')" "decode:$(grep -iE 'canary' $R/runs/canary_eager_$m/decode/server.log | grep -vE 'server_args|install_canary|attach_canary' | tail -n 3 | cut -c1-180 | tr '\n' '|')"; do echo "   $u"; done
    echo "   exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/canary_eager_$m/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception' $R/runs/canary_eager_$m/decode/server.log); health prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
  else log "$m pair did not come up"; echo "   last prefill: $($SSH_A "sudo -n grep -vE 'UserWarning|return func' /home/alisachen_google_com/pd_disagg/results/runs/canary_eager_$m/prefill/server.log | tail -n 2 | cut -c1-180" | tr '\n' '|')"; echo "   last decode: $(grep -vE 'UserWarning|return func' $R/runs/canary_eager_$m/decode/server.log | tail -n 2 | cut -c1-180 | tr '\n' '|')"; fi
done
bash $S/pd.sh stop >/dev/null 2>&1
$SSH_A "$GA checkout -q main && echo '   host a restored'"; cd $HOME/sglang-torchtpu && git checkout -q main && log "phase canary eager done"
