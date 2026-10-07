#!/usr/bin/env bash
# Print who holds the TPU on both hosts; exit 0 only if both are free of OTHER users' processes.
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 sa_101411256518727260534@10.128.15.217"
a=$($SSH_A "sudo -n fuser -v /dev/vfio/0 2>&1 | awk 'NR>1{print \$(NF-3)}' | sort -u | tr '\n' ' '" 2>/dev/null)
b=$(sudo -n fuser -v /dev/vfio/0 2>&1 | awk 'NR>1{print $(NF-3)}' | sort -u | tr '\n' ' ')
echo "$(date -u +%FT%TZ) a_holders=[${a}] b_holders=[${b}]"
me=$(id -un); for h in $a $b; do [[ $h != $me ]] && exit 1; done; exit 0
