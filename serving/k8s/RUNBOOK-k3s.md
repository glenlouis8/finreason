# Serve the 7B AWQ model on real Kubernetes (k3s + GPU)

Goal: the actual `glen-louis/finreason-qwen2.5-7b-awq` running in a K8s pod on a
GPU node, reachable through a Service. Produces the un-caveated screenshot:
"served AWQ Qwen2.5-7B on Kubernetes (k3s) with GPU."

Cost: ~$2-5, ~1 hr. Tear the box down when done.

## 1. Provision a GPU box
Any GPU cloud with an **Ubuntu + NVIDIA driver** image works. Lowest entry:
- TensorDock ($5 min) / Vast.ai / Jarvis Labs.
- Pick: **1× A10/A40/L4/3090 (24GB+)**, Ubuntu 22.04, ~50GB disk.
- Confirm drivers: `nvidia-smi` should show the GPU.

## 2. Install k3s (single-node real Kubernetes)
```bash
curl -sfL https://get.k3s.io | sh -
sudo k3s kubectl get nodes        # node Ready
# make kubectl usable without sudo:
mkdir -p ~/.kube && sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config && sudo chown $(id -u):$(id -g) ~/.kube/config
export KUBECONFIG=~/.kube/config
```

## 3. Let k3s see the GPU (NVIDIA device plugin)
```bash
# nvidia container toolkit (if image doesn't have it)
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart k3s

# device plugin daemonset — exposes nvidia.com/gpu to the scheduler
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml

# confirm the node now advertises a GPU:
kubectl get node -o jsonpath='{.items[0].status.allocatable}' | tr ',' '\n' | grep nvidia
# -> "nvidia.com/gpu":"1"
```

## 4. Deploy the model
```bash
git clone https://github.com/glenlouis8/finreason.git
kubectl apply -f finreason/serving/k8s/vllm-deployment.yaml
kubectl get pods -w            # Pending -> ContainerCreating -> Running (model loads ~1-2 min)
kubectl logs -f deploy/finreason-vllm   # watch vLLM load, "Application startup complete"
```

## 5. Hit it THROUGH K8s + screenshot
```bash
kubectl port-forward svc/finreason-vllm 8000:8000 &
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "glen-louis/finreason-qwen2.5-7b-awq",
  "messages": [{"role":"system","content":"You are a financial analyst. End with a line Final Answer:"},
               {"role":"user","content":"Revenue grew from $4.2B to $5.7B. Percent growth?"}]
}'
```

## Screenshots to capture (these go in the repo)
1. `nvidia-smi` — the GPU
2. `kubectl get nodes` + the `nvidia.com/gpu: 1` allocatable line
3. `kubectl get pods` — finreason-vllm **Running** on the node
4. `kubectl logs deploy/finreason-vllm` — vLLM "startup complete"
5. the `curl` returning a FinReason answer ending `Final Answer:`
6. `kubectl get svc` — the Service

Save them under `serving/docs/screenshots/`. Then **destroy the box** to stop billing.
```bash
# optional cleanup first
kubectl delete -f finreason/serving/k8s/vllm-deployment.yaml
```
