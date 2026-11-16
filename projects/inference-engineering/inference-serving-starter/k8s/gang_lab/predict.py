"""Offline check of the gang-scheduling lab. GIVEN. Needs only PyYAML, no cluster, no Docker, no kind.

    python k8s/gang_lab/predict.py

It does two things and states plainly which one is which.

1. VALIDATES the manifest: every Pod has the fields the scheduler reads (name, namespace, restartPolicy,
   a container with an image and an nvidia.com/gpu limit that is a whole number) and the gangs are what
   the README says they are. This is a structural check, not `kubectl --dry-run`: a real API server is
   needed for that.
2. PREDICTS the outcome with a MODEL of the scheduler: pods are placed one at a time in creation order, each
   on the first worker with enough free GPUs, and a pod that does not fit stays Pending. The real
   kube-scheduler scores nodes and may spread pods differently, but with a free-GPU count as the only
   constraint the SET of pods that fit depends on the order, not on which node holds them. It is a model
   of how the lab should behave. It is not evidence that it did: only `kubectl get pods` on a cluster is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
GPU = "nvidia.com/gpu"


def load_pods(path: Path) -> list:
    docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]
    pods = [d for d in docs if d.get("kind") == "Pod"]
    if not pods:
        raise ValueError("no Pod in the manifest")
    return pods


def validate(pods: list) -> list:
    """Returns (name, gang, gpus) per pod in creation order; raises ValueError on a bad manifest."""
    out = []
    for p in pods:
        name = p["metadata"]["name"]
        if p["apiVersion"] != "v1" or not p["metadata"].get("namespace"):
            raise ValueError(f"{name}: needs apiVersion v1 and a namespace")
        spec = p["spec"]
        if spec.get("restartPolicy") not in ("Never", "OnFailure", "Always"):
            raise ValueError(f"{name}: restartPolicy missing")
        (c,) = spec["containers"]
        if not c.get("image"):
            raise ValueError(f"{name}: container has no image")
        limit = (c.get("resources") or {}).get("limits", {}).get(GPU)
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{name}: {GPU} must be a whole-number limit (extended resources go in limits)")
        out.append((name, p["metadata"]["labels"]["gang"], limit))
    return out


def schedule(pods: list, nodes: int, gpus_per_node: int) -> dict:
    """FIFO placement model. Returns {"running": [...], "pending": [...], "free": [...]}."""
    free = [gpus_per_node] * nodes
    running, pending = [], []
    for name, _gang, need in pods:
        for i, f in enumerate(free):
            if f >= need:
                free[i] -= need
                running.append(name)
                break
        else:
            pending.append(name)
    return {"running": running, "pending": pending, "free": free}


def gang_sizes(pods: list) -> dict:
    sizes: dict = {}
    for _name, gang, need in pods:
        sizes[gang] = sizes.get(gang, 0) + need
    return sizes


def predict(path: Path = HERE / "gang-deadlock.yaml", nodes: int = 2, gpus_per_node: int = 3) -> dict:
    pods = validate(load_pods(path))
    sizes = gang_sizes(pods)
    res = schedule(pods, nodes, gpus_per_node)
    held: dict = {g: 0 for g in sizes}
    gang_of = {n: g for n, g, _ in pods}
    for n in res["running"]:
        held[gang_of[n]] += 1
    complete = [g for g in sizes if held[g] == sizes[g]]
    deadlock = not complete and bool(res["pending"]) and sum(res["free"]) < min(
        sizes[g] - held[g] for g in sizes)
    return {"gang_sizes": sizes, "held": held, "pending": res["pending"], "free_gpus": sum(res["free"]),
            "complete_gangs": complete, "deadlock": deadlock, "capacity": nodes * gpus_per_node}


def main() -> int:
    r = predict()
    print(f"manifest is well formed: {len(r['gang_sizes'])} gangs, sizes {r['gang_sizes']}, "
          f"cluster capacity {r['capacity']} GPUs")
    print(f"model of the scheduler (NOT a cluster run): holds {r['held']}, free {r['free_gpus']}, "
          f"pending {r['pending']}")
    print("predicted: DEADLOCK, no gang can complete" if r["deadlock"] else "predicted: no deadlock")
    return 0 if r["deadlock"] else 1


if __name__ == "__main__":
    sys.exit(main())
