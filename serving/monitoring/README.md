# Monitoring (Prometheus + Grafana)

Installs the `kube-prometheus-stack` Helm chart: Prometheus (collects metrics)
+ Grafana (graphs them) + pre-built Kubernetes dashboards, wired together.

## Install
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

kubectl create namespace monitoring
helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring

kubectl get pods -n monitoring -w   # wait until all Running
```

## Open Grafana
```bash
# get the auto-generated admin password (stored in a K8s Secret)
kubectl get secret -n monitoring monitoring-grafana \
  -o jsonpath="{.data.admin-password}" | base64 -d; echo

kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80
```
Browser → http://localhost:3000 — user `admin`, password from above.

## See the served pods
Dashboards → **Kubernetes / Compute Resources / Namespace (Pods)** →
set namespace = `default` → live CPU/memory of the `finreason-web` pods.

Chain: kubelet → metrics → Prometheus → Grafana. Same numbers the HPA scales on.

## Money screenshot
Run the Locust load test with this dashboard open → watch CPU climb and pods
multiply on the graph in real time. That GIF/screenshot goes in the top README.
