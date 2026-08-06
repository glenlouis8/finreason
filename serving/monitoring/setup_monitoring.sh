#!/usr/bin/env bash
# Stand up Prometheus + Grafana against a running vLLM server, in one shot.
#
#   bash serving/monitoring/setup_monitoring.sh <grafana-public-host>
#
# e.g. on RunPod, where port 8888 is the pre-exposed HTTP port:
#   bash serving/monitoring/setup_monitoring.sh fa0eoiplsh4q5h-8888.proxy.runpod.net
#
# Why the host argument: Grafana rejects requests whose Origin doesn't match its
# configured root_url ("origin not allowed"). Behind RunPod's proxy the browser's
# origin is the proxy domain, not localhost, so it must be declared up front.
# CLI `cfg:` overrides did not reliably apply — a real custom.ini does.
#
# Data source AND dashboard are provisioned from files. Creating them through the
# UI needs POSTs that hit the same CSRF path, so file provisioning avoids the
# problem entirely rather than fighting it.
set -euo pipefail

GF_HOST="${1:?usage: setup_monitoring.sh <grafana-public-host>}"
PROM_VER="3.1.0"
GRAF_VER="11.5.0"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Install to local disk, not the network volume: Grafana is ~10k small files and
# untarring it onto a FUSE mount takes minutes instead of seconds.
INSTALL_DIR="${INSTALL_DIR:-/root/monitoring}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "==> checking vLLM is up and exposing metrics"
curl -sf http://localhost:8000/health > /dev/null || {
  echo "vLLM not responding on :8000 — start it first" >&2; exit 1; }
curl -s http://localhost:8000/metrics | grep -q "^vllm:" || {
  echo "no vllm: metrics found on :8000/metrics" >&2; exit 1; }

echo "==> Prometheus $PROM_VER"
if [ ! -d "prometheus-${PROM_VER}.linux-amd64" ]; then
  wget -q "https://github.com/prometheus/prometheus/releases/download/v${PROM_VER}/prometheus-${PROM_VER}.linux-amd64.tar.gz"
  # --no-same-owner: FUSE-backed volumes reject chown and abort the extraction.
  tar --no-same-owner -xzf "prometheus-${PROM_VER}.linux-amd64.tar.gz"
fi

cat > prometheus.yml <<EOF
global:
  scrape_interval: 2s
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ['localhost:8000']
EOF

pkill -f "prometheus --config" 2>/dev/null || true
(cd "prometheus-${PROM_VER}.linux-amd64" && \
  nohup ./prometheus --config.file="${INSTALL_DIR}/prometheus.yml" \
    --web.listen-address=:9090 > "${INSTALL_DIR}/prom.log" 2>&1 &)
sleep 5
curl -sf "http://localhost:9090/api/v1/query?query=up" > /dev/null \
  && echo "    prometheus ok on :9090"

echo "==> Grafana $GRAF_VER"
if [ ! -d "grafana-v${GRAF_VER}" ]; then
  wget -q "https://dl.grafana.com/oss/release/grafana-${GRAF_VER}.linux-amd64.tar.gz"
  tar --no-same-owner -xzf "grafana-${GRAF_VER}.linux-amd64.tar.gz"
fi
cd "grafana-v${GRAF_VER}"

cat > conf/custom.ini <<EOF
[server]
http_port = 8888
domain = ${GF_HOST}
root_url = https://${GF_HOST}/

[security]
csrf_trusted_origins = ${GF_HOST} https://${GF_HOST}
csrf_additional_headers = X-Forwarded-Host
cookie_secure = true
cookie_samesite = none

[auth.anonymous]
enabled = true
org_role = Viewer

[users]
default_theme = dark
EOF

mkdir -p conf/provisioning/datasources conf/provisioning/dashboards
cat > conf/provisioning/datasources/prometheus.yaml <<'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
EOF

cat > conf/provisioning/dashboards/finreason.yaml <<'EOF'
apiVersion: 1
providers:
  - name: finreason
    type: file
    options:
      path: /var/lib/grafana/dashboards
EOF

mkdir -p /var/lib/grafana/dashboards
cp "${REPO_DIR}/serving/monitoring/grafana_dashboard.json" /var/lib/grafana/dashboards/

# Port 8888 is RunPod's pre-exposed HTTP port (nominally Jupyter). Reusing it
# avoids editing the pod to open a new port, which would restart it and kill vLLM.
pkill -f jupyter 2>/dev/null || true
pkill -f "grafana server" 2>/dev/null || true
nohup ./bin/grafana server --homepath . > "${INSTALL_DIR}/grafana.log" 2>&1 &
sleep 12

if curl -sf -o /dev/null http://localhost:8888/login; then
  echo "    grafana ok on :8888"
else
  echo "    grafana failed to start — see ${INSTALL_DIR}/grafana.log" >&2; exit 1
fi

cat <<EOF

==> ready
  Dashboard:  https://${GF_HOST}/d/finreason-vllm
  Login:      admin / admin   (anonymous viewing also enabled)

Generate load so the panels have something to show:
  python serving/monitoring/capture_metrics.py --load 40 --interval 5 --duration 900
EOF
