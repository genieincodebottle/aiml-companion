#!/usr/bin/env bash
# Advertise a FAKE extended resource, nvidia.com/gpu, on every worker node of the kind cluster.
# kind has no GPUs, so this is bookkeeping only: the scheduler will count the GPUs and place pods
# against them, and nothing will ever run on a real device. That is exactly what the gang-scheduling
# lab needs, because the deadlock is a scheduling decision, not a GPU one.
#
# Method: the Kubernetes docs "Advertise Extended Resources for a Node" patch /status/capacity through
# `kubectl proxy`, with "/" in the resource name written as "~1" in the JSON Patch path
# (https://kubernetes.io/docs/tasks/administer-cluster/extended-resource-node/, read 2026-11-16).
# The value must be an integer. Run `bash fake-gpus.sh remove` to take the resource off again.
#
#   GPUS_PER_NODE=3 bash k8s/gang_lab/fake-gpus.sh
set -euo pipefail
CLUSTER_CONTEXT="${CLUSTER_CONTEXT:-kind-serving}"
GPUS_PER_NODE="${GPUS_PER_NODE:-3}"
ACTION="${1:-add}"
PORT="${PROXY_PORT:-8001}"

kubectl --context "$CLUSTER_CONTEXT" proxy --port "$PORT" >/dev/null 2>&1 &
PROXY_PID=$!
trap 'kill $PROXY_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
  curl -fs "http://localhost:${PORT}/api" >/dev/null 2>&1 && break
  sleep 0.5
done

# Workers only: the control-plane node is tainted and should not hold gang members.
NODES=$(kubectl --context "$CLUSTER_CONTEXT" get nodes -l '!node-role.kubernetes.io/control-plane' \
  -o jsonpath='{.items[*].metadata.name}')
[ -n "$NODES" ] || { echo "no worker nodes found in context $CLUSTER_CONTEXT" >&2; exit 1; }

for NODE in $NODES; do
  if [ "$ACTION" = "remove" ]; then
    BODY='[{"op": "remove", "path": "/status/capacity/nvidia.com~1gpu"}]'
  else
    BODY="[{\"op\": \"add\", \"path\": \"/status/capacity/nvidia.com~1gpu\", \"value\": \"${GPUS_PER_NODE}\"}]"
  fi
  curl -fsS --header "Content-Type: application/json-patch+json" --request PATCH \
    --data "$BODY" "http://localhost:${PORT}/api/v1/nodes/${NODE}/status" >/dev/null
  echo "${ACTION}: ${NODE} nvidia.com/gpu=${GPUS_PER_NODE}"
done
echo "check:  kubectl --context $CLUSTER_CONTEXT describe nodes | grep -A1 'nvidia.com/gpu'"
