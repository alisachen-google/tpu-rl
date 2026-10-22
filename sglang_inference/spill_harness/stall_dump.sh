#!/usr/bin/env bash
# Dump the top frames of every scheduler rank on both engines (hang diagnosis).
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
echo "### prefill (host a) $(date -u +%T)"
$SSH_A "sudo -n bash -c 'for P in \$(ps -eo pid,cmd | grep -E \"sglang::scheduler_TP\" | grep -v grep | sort -k2 | awk \"{print \\\$1}\"); do N=\$(ps -o cmd= -p \$P | sed \"s/.*scheduler_//\"); echo \"== prefill \$N\"; env PATH=/home/alisachen_google_com/venv/sglang/bin:\$PATH py-spy dump --pid \$P 2>&1 | grep -A 7 MainThread | sed -n 2,8p | sed \"s/^ *//\" | cut -c1-110; done'" 2>&1
echo "### decode (host b) $(date -u +%T)"
for P in $(ps -eo pid,cmd | grep -E "sglang::scheduler_TP" | grep -v grep | sort -k2 | awk '{print $1}'); do N=$(ps -o cmd= -p $P | sed 's/.*scheduler_//'); echo "== decode $N"; sudo -n env PATH=$HOME/venv/sglang/bin:$PATH py-spy dump --pid $P 2>&1 | grep -A 7 MainThread | sed -n 2,8p | sed 's/^ *//' | cut -c1-110; done
