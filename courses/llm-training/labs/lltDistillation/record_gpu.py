"""Turn one notebook kernel run (distill_results.json + notebook_stdout.txt) into Run Library files. No new measurement.

  python record_gpu.py <kernel output dir> <tag> "<kernel note>"
tag main     -> lltDistillation-T2-baselines, -T2-teacher-data, one file per trained arm, -T2-trl-distill
tag notebook -> lltDistillation-T2-notebook (the learner defaults, with USE_OUR_ADAPTER = True)
tag smoke    -> lltDistillation-T2-smoke (tiny sizes; timing of the teacher's generation)
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import write_run  # noqa: E402

out, tag, note = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
R = json.load(open(out / "distill_results.json", encoding="utf-8"))
txt = (out / "notebook_stdout.txt").read_text(encoding="utf-8")
cells = {m.group(1): float(m.group(2)) for m in re.finditer(r"CELL (cell[0-9]+_step[0-9]+) end ([0-9.]+)", txt)}
peak = re.search(r"KERNEL peak GB over the whole session ([0-9.]+)", txt)
ver = re.search(r"Python ([0-9.]+) [|] GPU ([^|]+)[|] torch ([^ ]+)", txt)
env = {"python": ver.group(1) if ver else None, "torch": ver.group(3) if ver else None, "transformers": "5.22.1", "trl": "1.17.1",
       "peft": "0.21.2", "accelerate": "1.15.0", "datasets": "5.1.0", "bitsandbytes": "0.50.2", "tokenizers": "0.23.3",
       "huggingface_hub": "2.4.0", "source": "notebook install cell pins; Python and torch from the runtime line of the log"}
hw = {"gpu": "Tesla T4", "count": 1, "vramGB": 15.0}
common = {"teacher": "Qwen/Qwen3-4B-Base, NF4 double quant, fp16 compute, frozen",
          "student": "Qwen/Qwen3-0.6B-Base, frozen fp16 base + fp32 LoRA r16 alpha 32 all-linear",
          "data": "openai/gsm8k main; test first N_TEST, train shuffled seed 0 first N_TRAIN",
          "prompt": "Question: <q> + newline + Answer:", "stop": "newline + Question:", "eval": "greedy, 256 new tokens"}


def strip(r):
    return {k: v for k, v in r.items() if k not in ("texts",)}


if tag == "main":
    arms = [k for k in R if k not in ("teacher", "student, before training")]
    write_run("lltDistillation-T2-baselines", "T2", "baseline", dict(common, kernel=note), 0, [],
              evals={"teacher": strip(R["teacher"]), "student": strip(R["student, before training"]),
                     "sampleTexts": {"teacher": R["teacher"]["texts"][:20], "student": R["student, before training"]["texts"][:20]},
                     "cellEndSec": cells}, wall_clock_sec=cells.get("cell04_step3"), hardware=hw, env=env)
    gen = re.search(r"teacher wrote ([0-9]+) tokens in ([0-9]+) s [|] ([0-9]+) tokens/s", txt)
    tline = re.search(r"teacher answers ([0-9]+) [|] correct ([0-9.]+) [|] ended at the stop string ([0-9.]+) [|] mean tokens ([0-9.]+)", txt)
    write_run("lltDistillation-T2-teacher-data", "T2", "baseline", dict(common, kernel=note, batch=64), 0, [],
              evals={"newTokens": int(gen.group(1)), "sec": int(gen.group(2)), "tokensPerSec": int(gen.group(3)),
                     "answers": int(tline.group(1)), "correct": float(tline.group(2)), "stopped": float(tline.group(3)),
                     "meanTokens": float(tline.group(4))}, wall_clock_sec=int(gen.group(2)), hardware=hw, env=env)
    for name in arms:
        r = R[name]
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        seed = 1 if name.endswith("seed 1") else 0
        write_run("lltDistillation-T2-" + slug, "T2", "ablation:arm=" + slug, dict(common, arm=name, kernel=note, steps=64, batch=32,
                  micro=4, lr=2e-4, warmup=3, schedule="cosine"), seed,
                  [{"step": i + 1, "trainLoss": v} for i, v in enumerate(r["loss"])],
                  evals=dict(strip(r), sampleTexts=r["texts"][:20]), wall_clock_sec=r["trainSec"], hardware=hw, env=env)
    trl = re.findall(r"[{]'step': [^}]*[}]", txt)
    tsec = re.search(r"3 steps in ([0-9.]+) s", txt)
    dt = re.search(r"trainable dtypes ([{][^}]*[}])", txt)
    write_run("lltDistillation-T2-trl-distill", "T2", "baseline", dict(common, kernel=note, trainer="trl.DistillationTrainer beta 1",
              steps=3, batch=8, maxCompletion=256), 0, [], evals={"log": trl, "sec": float(tsec.group(1)) if tsec else None,
              "trainableDtypes": dt.group(1) if dt else None}, hardware=hw, env=env)
else:
    write_run("lltDistillation-T2-" + tag, "T2", "baseline", dict(common, kernel=note), 0, [],
              evals={"results": {k: strip(v) for k, v in R.items()}, "cellEndSec": cells, "kernelPeakGB": float(peak.group(1)) if peak else None,
                     "stdout": txt}, wall_clock_sec=max(cells.values()) if cells else None, hardware=hw, env=env)
print("cells", cells, "| peak", peak.group(1) if peak else None)