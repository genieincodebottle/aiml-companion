"""Write the lltQLoRA Run Library files.

  python make_runs.py broken            three broken-run files from earlier Kaggle runs (sources named inside)
  python make_runs.py lab <json> <log>  run files from the llt-qlora-lab kernel output (llt_qlora_lab.json)
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
RUNS = os.path.join(ROOT, "runs", "lltQLoRA")
os.makedirs(RUNS, exist_ok=True)


def write(run):
    json.dump(run, open(os.path.join(RUNS, run["runId"] + ".json"), "w"), indent=1)
    print("wrote", run["runId"])


def broken():
    probe = json.load(open(os.path.join(ROOT, "runs", "_probes", "llt-model-probe-2026-11.json")))
    byTag = {r["tag"]: r for r in probe["runs"]}
    a, b = byTag["q35-9b-qlora"], byTag["q35-9b-qlora-v2"]
    common = {"module": "lltQLoRA", "tier": "T2", "hardware": {"gpu": "Tesla T4", "count": 1, "vramGB": 15.6},
              "env": probe["env"], "date": probe["date"], "costUSD": 0}
    write(dict(common, runId="lltQLoRA-T2-broken-kbit-prep-v2026-11", label="broken:prepare_model_for_kbit_training",
               source="runs/_probes/llt-model-probe-2026-11.json, run q35-9b-qlora (kernel llt-model-probe v2)",
               config={"model": a["model"], "quant": "nf4, double quant, fp16 compute", "prep": "prepare_model_for_kbit_training"},
               error=a["error"], peakMemGB=a["peakMemGB"], loadSec=a["loadSec"],
               derivation="embedding 248,320 x 4,096 = 1,017,118,720 weights x 4 bytes = 4,068,474,880 B = 3.79 GiB"))
    write(dict(common, runId="lltQLoRA-T2-broken-skipped-steps-v2026-11", label="broken:gradscaler-skips-all-steps",
               source="runs/_probes/llt-model-probe-2026-11.json, run q35-9b-qlora-v2 (kernel llt-model-probe v3)",
               config={"model": b["model"], "batch": "2 x 1,024", "steps": b["stepsDone"], "gradScalerInit": 65536},
               metrics=[{"step": i + 1, "trainLoss": l, "sec": t} for i, (l, t) in enumerate(zip(b["losses"], b["stepTimes"]))],
               skippedSteps=b["skippedSteps"], gradScalerFinal=b["gradScalerFinal"], peakMemGB=b["peakMemGB"],
               note="every optimizer step was skipped (5 of 5); the loss differs only because each step saw a new batch"))
    log = open(os.path.join(ROOT, "kaggle", "out", "llt-finetune-test", "llt-finetune-test.log"), encoding="utf8").read()
    m = re.search(r'NotImplementedError: [^"]*?"_amp_foreach_non_finite_check_and_unscale_cuda[^\n]*?BFloat16', log)
    write(dict(common, runId="lltQLoRA-T2-broken-bf16-adapters-v2026-11", label="broken:sftrainer-bf16-adapters",
               source="kaggle/out/llt-finetune-test/llt-finetune-test.log (QLoRA cell, Qwen3-8B-Base, SFTTrainer peft_config) and "
                      "kaggle/out/llt-qlora-test/llt_qlora_results.json (504 trainable tensors bf16 after SFTTrainer init)",
               env={"python": "3.13.16", "torch": "2.11.0+cu130", "transformers": "5.22.1", "trl": "1.17.1", "peft": "0.21.2"},
               date="2026-11-05",
               error='NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for \'BFloat16\'',
               errorFoundInLog=bool(m),
               cause="trl/trainer/sft_trainer.py (1.17.1): for a quantized model, every trainable parameter except DoRA magnitudes is cast to bf16 in __init__"))


def lab(jpath, logpath=None, run_id="lltQLoRA-T2-lab-v2026-11", label="baseline", kernel_version=3, note=None):
    R = json.load(open(jpath))
    common = {"module": "lltQLoRA", "tier": "T2", "hardware": {"gpu": R["gpu"], "count": 1, "vramGB": 15.6},
              "env": R["env"], "date": R["date"], "costUSD": 0,
              "notebook": "labs/lltQLoRA/lltQLoRA.ipynb, run as Kaggle kernel llt-qlora-lab v" + str(kernel_version)}
    body = {k: R[k] for k in R if k not in ("env", "date", "gpu", "gpuCount", "cellSec")}
    run = dict(common, runId=run_id, label=label, **body)
    if note:
        run["note"] = note
    write(run)


def int4_sweep(tryit_json, kaggle_log):
    """Add the uniform-int4 block-size sweep to lltQLoRA-t1-nf4-v2026-11.json from its two measured sources."""
    t = json.load(open(tryit_json))["blockSweep"]
    L = json.load(open(kaggle_log, encoding="utf8"))
    line = [x["data"] for x in L if x["data"].startswith("int4 relative error % at block 64, 4096, one row, whole tensor")][0]
    printed = json.loads(line.split(":", 1)[1].strip())
    p = os.path.join(RUNS, "lltQLoRA-t1-nf4-v2026-11.json")
    run = json.load(open(p))
    run["int4BlockSweep"] = {
        "relErr": {"64": t["64"]["int4"], "256": t["256"]["int4"], "4096": t["4096"]["int4"], "12288": t["12288"]["int4"],
                   "50331648": printed[3] / 100},
        "printedByNotebookCell5Percent": {"64": printed[0], "4096": printed[1], "12288": printed[2], "50331648": printed[3]},
        "sources": {"64, 256, 4096, 12288": "measure.py tryit on this laptop (T1, numpy 2.5.4), blockSweep in llt5/lltQLoRA.tryit_data.json",
                    "50331648 and the printed values": "kaggle/out/llt-qlora-lab/llt-qlora-lab.log, notebook code cell 5 output (kernel v3)"},
        "note": "0.17585 at block 4,096 prints as 17.58 in the cell (round to 2 decimals); the page uses the printed 17.58."}
    json.dump(run, open(p, "w"), indent=1)
    print("int4 sweep added", run["int4BlockSweep"]["relErr"])


if __name__ == "__main__":
    if sys.argv[1] == "broken":
        broken()
    elif sys.argv[1] == "lab":
        lab(sys.argv[2])
    elif sys.argv[1] == "int4-sweep":
        int4_sweep(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "lab-partial":
        lab(sys.argv[2], run_id="lltQLoRA-T2-lab-run2-partial-v2026-11", label="replicate:steps-1-to-9", kernel_version=2,
            note="Kernel v2 stopped with an out-of-memory error in step 10 (dev_loss batch 64 on the proxy). The memory sweep "
                 "printed out of memory for every shape because the model was still in eval mode, so gradient checkpointing "
                 "was off. Steps 2-9 are a second measurement of the same cells as lltQLoRA-T2-lab-v2026-11.")
