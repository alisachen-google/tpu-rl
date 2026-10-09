#!/usr/bin/env bash
# Start (or ensure) the Prometheus + Grafana stack for the PD pair. Idempotent; safe to call from pd.sh start.
cd "$(dirname "$0")" || exit 1
python3 build_dashboard.py >/dev/null 2>&1 || true
sudo -n docker compose up -d --remove-orphans >/dev/null 2>&1 || { echo "monitoring: docker compose failed" >&2; exit 1; }
echo "monitoring: grafana http://$(hostname -I | awk '{print $1}'):3000/d/sglang-pd  prometheus :9090"
