#!/usr/bin/env bash
# Sum the reverse-pull commits recorded in the prefill server log on host a for a run.
RUN=${1:?run}; SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
$SSH "sudo -n grep -h 'peer prefix committed\|pull at schedule time' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log 2>/dev/null" \
 | python3 -c "
import re,sys
n=0; tok=0; sched=0
for line in sys.stdin:
    if 'schedule time' in line:
        sched+=1; continue
    m=re.search(r'matched=(\d+) hit=(\d+)', line)
    if m: n+=1; tok+=int(m.group(2))-int(m.group(1))
print(f'peer_commits={n} peer_tokens={tok} schedule_time_pulls={sched}')"
