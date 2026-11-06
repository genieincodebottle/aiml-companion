"""Copy the Kaggle outputs of lltInstructionData into the Run Library.

Usage:
  python record_gpu.py teacher <kaggle out dir of llt-insdata-teacher>
  python record_gpu.py notebook <kaggle out dir of llt-insdata-notebook>
  python record_gpu.py exp <kaggle out dir of llt-insdata-exp>
"""
import json
import sys
from pathlib import Path

from runlog import write_run

T4 = {"gpu": "Tesla T4", "count": 1, "vramGB": 15.6}


def kaggle_log_text(out_dir):
    logs = list(Path(out_dir).glob("*.log"))
    if not logs:
        return ""
    try:
        return "".join(x["data"] for x in json.loads(logs[0].read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return logs[0].read_text(encoding="utf-8", errors="replace")


def teacher(out_dir):
    R = json.loads((Path(out_dir) / "insdata_teacher.json").read_text(encoding="utf-8"))
    nl = chr(10)
    metrics = []
    for name, T in R["teachers"].items():
        qs = T["instructions"]
        metrics.append({
            "teacher": name, "loadedClass": T["class"], "loadSec": T["loadSec"],
            "instrTokPerSec": T["instrTokPerSec"], "answerTokPerSec": T["answerTokPerSec"],
            "instrSec": T["instrSec"], "answerSec": T["answerSec"], "peakMemGB": T["peakMemGB"],
            "usableShare": T["usableShare"], "answersEnded": T["answersEnded"],
            "runOnShare": sum((nl + nl) in q for q in qs) / len(qs),
            "medianInstructionChars": sorted(len(q) for q in qs)[len(qs) // 2],
            "medianReward": T["medianReward"], "meanCosine": T["meanCosine"], "vendi": T["vendi"],
        })
    samples = [{"teacher": n, "instruction": q[:300]} for n, T in R["teachers"].items() for q in T["instructions"][:10]]
    write_run("lltInstructionData-T2-teacher-bakeoff", "T2", "ablation:teacher", {
        "candidates": list(R["teachers"]), "requests": 128, "instructionMaxNew": 128, "answerMaxNew": 512,
        "sampling": "temperature 1.0, top_p 1.0, top_k 0 for the request; greedy answer, thinking off",
        "batch": "64 (32 for answers of the 4B model)", "dtype": "float16",
        "judge": "Skywork/Skywork-Reward-V2-Qwen3-0.6B", "embedder": "ibm-granite/granite-embedding-311m-multilingual-r2",
        "usableRule": "15 to 1,500 characters, no <think>, does not start with a role name, langdetect says en",
        "kernel": "llt-insdata-teacher v2"}, 0, metrics, samples=samples, hardware=T4,
        env=dict(R["env"], kaggleImage="2026-11-06"))


if __name__ == "__main__":
    {"teacher": teacher}[sys.argv[1]](sys.argv[2])
