#!/usr/bin/env bash
# Kill OUR servers only (never another user's), remove stale lock/shm files owned by us.
me=$(id -un)
pkill -u "$me" -f 'sglang.launch_server' ; pkill -u "$me" -f 'sglang::' ; pkill -u "$me" -f 'sglang_router' ; sleep 3
pkill -9 -u "$me" -f 'sglang.launch_server' ; pkill -9 -u "$me" -f 'sglang::' ; pkill -9 -u "$me" -f 'sglang_router' ; true
find /dev/shm -maxdepth 1 -user "$me" \( -name 'sgl*' -o -name 'sem.*' -o -name 'torch*' \) -delete 2>/dev/null
[ -O /tmp/libtpu_lockfile ] && rm -f /tmp/libtpu_lockfile
echo "[cleanup] remaining sglang procs (all users): $(pgrep -fc 'sglang.launch_server|sglang::scheduler')  TPU holders: $(sudo -n fuser -v /dev/vfio/0 2>&1 | awk 'NR>1{print $(NF-3)}' | sort -u | tr '\n' ' ')"
