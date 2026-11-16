# Gang-scheduling lab: make the deadlock happen (kind, no GPU)

**Optional and not gated by `make check`.** A multi-GPU serving job (tensor parallel 4, say) needs
all of its ranks running at once. The default Kubernetes scheduler places pods one at a time and
does not know that. Give it two such jobs and a cluster that fits one and a half, and each job grabs
part of the GPUs and waits forever for the rest. This lab reproduces that on a laptop by telling the
scheduler the kind workers have GPUs they do not have.

## Honest status

**Not executed here.** When this lab was written, Docker was available but `kind` was not installed, so the
cluster steps below were never run, and nothing in this file is an observed result. What was run:

| Check | Command | Result (2026-11-16) |
|---|---|---|
| Manifest structure: names, namespace, whole-number GPU limits, gang sizes | `python k8s/gang_lab/predict.py` | well formed, 2 gangs of 4 GPUs |
| A model of the scheduler (FIFO, first worker with room) on 2 workers x 3 GPUs | same script | each gang holds 3 of 4, `a3` and `b3` pending, deadlock |
| Shell syntax of `fake-gpus.sh` | `bash -n` | ok |
| The fake-GPU patch method | read from the Kubernetes docs (link in `fake-gpus.sh`) | not executed |

The model is a model: only `kubectl get pods` on your cluster shows what the real scheduler did. If your
run disagrees with it (for example both gangs fit, or a different set of pods is Pending), that is a finding
worth a paragraph. Say in your notes which of the steps below you ran.

You need: Docker, [kind](https://kind.sigs.k8s.io/), kubectl, and the `engine-sim:dev` image from
`k8s/RUNBOOK.md` step 1 (the pods reuse it so nothing is pulled).

## 1. A cluster with fake GPUs

```bash
kind create cluster --name serving --config k8s/kind-config.yaml     # 1 control plane + 2 workers
docker build -f k8s/Dockerfile -t engine-sim:dev .
kind load docker-image engine-sim:dev --name serving
GPUS_PER_NODE=3 bash k8s/gang_lab/fake-gpus.sh                       # 2 workers x 3 = 6 fake GPUs
kubectl describe nodes | grep -A1 nvidia.com/gpu                     # Capacity and Allocatable show 3
```

`fake-gpus.sh` patches `/status/capacity/nvidia.com~1gpu` through `kubectl proxy`, the method the
Kubernetes docs give for advertising an extended resource. The kubelet owns that field, so if a node restarts,
run the script again. `bash k8s/gang_lab/fake-gpus.sh remove` takes it off.

## 2. Reproduce the deadlock

```bash
kubectl apply -f k8s/gang_lab/gang-deadlock.yaml
kubectl -n gang-lab get pods -L gang -o wide
kubectl -n gang-lab describe pod a3        # the Events section says why it is Pending
```

Two jobs, `a` and `b`, of four pods each, one GPU per pod, created in the order a0 b0 a1 b1 a2 b2 a3 b3.
**Expected, from how the scheduler works:** six pods Running (three of each job), `a3` and `b3` Pending with a
FailedScheduling event that names `nvidia.com/gpu` as insufficient. The exact event wording depends on your
Kubernetes version. All 6 GPUs are held and no job can make progress; nothing ever times out or restarts.

## 3. Questions to answer in your own notes

1. How many GPUs are idle? How many are useless? Why is the second number the one that costs money?
2. Delete `b0` through `b3` and watch `a3` start. Which order of deletes frees a whole gang soonest? What does that
   say about preemption as a fix?
3. Would a higher `priorityClassName` on job `a` fix it? What does preemption do to the work `b` has already done, and
   what happens if `b` is rescheduled while `a` is still starting?
4. What would you want from a scheduler for this workload? Write it as one rule, then check it against the options below.

## 4. The fix: admit a whole gang or none of it

Install one gang-aware layer and rewrite the two jobs for it. Three common choices:
[Kueue](https://kueue.sigs.k8s.io/), [Volcano](https://volcano.sh/) and the scheduler-plugins coscheduling
plugin. This repo does not ship a fix manifest, because the lab was not run and the APIs move between releases.

Facts read on 2026-11-16 that help you start with Kueue (check the docs for the version you install):

* A Job joins a queue through the label `kueue.x-k8s.io/queue-name: <local-queue>`. Kueue suspends the Job
  and starts it when its quota admits it, instead of letting pods trickle onto nodes
  (https://kueue.sigs.k8s.io/docs/tasks/run/jobs/).
* The ClusterQueue example in the docs uses `apiVersion: kueue.x-k8s.io/v1beta2` with `resourceGroups`,
  `coveredResources`, `flavors` and `nominalQuota` (https://kueue.sigs.k8s.io/docs/concepts/cluster_queue/).
  You add `nvidia.com/gpu` to `coveredResources` with `nominalQuota: 6`. The docs pages read did not show a
  ResourceFlavor or LocalQueue example, so take their apiVersion from the docs for your release.
* GitHub listed v0.20.6 as the latest release on 2026-11-16 (https://github.com/kubernetes-sigs/kueue/releases).

**What to see when it works:** `a` runs to completion on 4 GPUs while `b` waits with 0 pods holding GPUs, then `b`
runs. The point is that no GPU sits held and useless while a job waits for its missing ranks.

## Tear down

```bash
kubectl delete namespace gang-lab
bash k8s/gang_lab/fake-gpus.sh remove
kind delete cluster --name serving
```
