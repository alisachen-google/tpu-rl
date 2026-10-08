#!/usr/bin/env bash
# Wait until nobody else holds the TPU on either host, then run the two-slot control pair.
S=$HOME/pd_disagg/scripts
until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
echo "$(date -u +%FT%TZ) TPU free again; starting control pair" >> $HOME/pd_disagg/results/phase_F11_control_mr2.log
exec bash $S/_phase_F11_control_mr2.sh >> $HOME/pd_disagg/results/phase_F11_control_mr2.log 2>&1
