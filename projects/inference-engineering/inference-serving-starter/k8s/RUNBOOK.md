# Stretch lab: autoscale on queue depth (kind, no GPU)

**Optional and not gated by `make check`.** The point is to see an autoscaler react to the signal
that matters for LLM serving, the number of requests waiting, using the same `/metrics` an
operator would have in production.

**Honest status:** the manifests are parsed by a test in this repo (valid YAML, the HPA metric name
matches the adapter rule) but they were **not run against a cluster in CI**. Chart versions and
the Prometheus service name below depend on the versions you install. If a command fails, read
the error: finding the mismatch is part of the exercise, and a short note on it makes a good
portfolio line.

You need: Docker, [kind](https://kind.sigs.k8s.io/), kubectl, Helm.

## 1. Cluster and image

```bash
kind create cluster --name serving --config k8s/kind-config.yaml
docker build -f k8s/Dockerfile -t engine-sim:dev .
kind load docker-image engine-sim:dev --name serving
```

## 2. Prometheus and the adapter

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install kps prometheus-community/kube-prometheus-stack -n monitoring --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
kubectl apply -f k8s/engine-sim.yaml
helm install adapter prometheus-community/prometheus-adapter -n monitoring -f k8s/prometheus-adapter-values.yaml
```

Check each layer before moving on. Most failures are one layer silently feeding the next an empty answer.

```bash
kubectl -n serving get pods                                       # engine-sim Running
kubectl -n serving port-forward svc/engine-sim 8100:8100 &        # then: curl localhost:8100/metrics | grep waiting
kubectl -n monitoring port-forward svc/kps-kube-prometheus-stack-prometheus 9090 &
#   open http://localhost:9090 and run:  vllm:num_requests_waiting   (needs a few scrapes)
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/serving/pods/*/vllm_num_requests_waiting"
```

If the last command returns an empty list, the adapter rule did not match the series: compare
the labels Prometheus shows (`namespace`, `pod`) with the rule's `resources.overrides`.

## 3. HPA and load

```bash
kubectl apply -f k8s/hpa.yaml
kubectl -n serving get hpa engine-sim -w &
python -m bench.cli open --base-url http://127.0.0.1:8100 --name k8s_load --qps 4 --duration 600 --out results/k8s_load.json
```

`port-forward` pins you to one pod, so the load does not spread. For a fair test, run the
generator inside the cluster against `http://engine-sim.serving.svc:8100`, or put an Ingress in
front. Think about why this matters for LLM traffic before you decide it is only a detail.

**What to see:** `num_requests_waiting` rises on the pod, the HPA's `TARGETS` column passes
`4`, replicas grow by up to 2 per 15 s, the waiting count per pod falls. Then stop the load and
watch the 120 s scale-down window.

## 4. Questions to answer in your own notes

1. A new pod starts with an empty prefix cache and, on a real GPU, 1 to 5 minutes of model load.
   What does that do to the scale-up you just watched? What would you change (warm pool, KV
   offload, a smaller scale-up step, scaling earlier)?
2. Queue depth is a trailing signal. Name a leading one, and say why you would or would not trust it.
3. Why average over pods (`AverageValue`) instead of summing? What breaks when one replica is hot
   and the rest idle (`debug_lab` incident B)?
4. What happens if the adapter returns no value for a pod that is still starting?

## Tear down

```bash
kind delete cluster --name serving
```
