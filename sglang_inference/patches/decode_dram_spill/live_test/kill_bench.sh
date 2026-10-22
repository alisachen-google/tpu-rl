#!/usr/bin/env bash
for p in $(pgrep -u $(id -un) -f "phase_dram_resu""me|spill_ben""ch.py|spill_stre""ss.py"); do kill $p 2>/dev/null && echo "killed $p"; done
