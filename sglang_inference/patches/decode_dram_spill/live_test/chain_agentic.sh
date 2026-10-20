#!/usr/bin/env bash
D=$HOME/pd_disagg/spill
until grep -q 'dram phase done' $D/logs/phase_dram_resume_spill_dram_4.log 2>/dev/null || ! pgrep -f 'phase_dram_resum[e].sh spill_dram_4' >/dev/null; do sleep 60; done
bash $D/phase_agentic.sh > $D/logs/phase_agentic.log 2>&1
echo "[chain-agentic] exit $?"
