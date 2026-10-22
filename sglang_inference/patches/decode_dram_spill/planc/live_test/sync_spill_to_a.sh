#!/usr/bin/env bash
# Mirror ~/sglang-spill (this host) to host a's ~/sglang-spill via the SA user.
set -uo pipefail
ME=alisachen_google_com; A=10.128.15.217
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no"
rsync -a --delete --partial --inplace -e "$SSH" --rsync-path='sudo rsync' \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude 'BASE_COMMIT.txt' \
  "$HOME/sglang-spill/" "sa_101411256518727260534@$A:/home/$ME/sglang-spill/" \
  && $SSH sa_101411256518727260534@$A "sudo -n chown -R $ME:$ME /home/$ME/sglang-spill" \
  && echo "synced $(date -u '+%F %T')"
