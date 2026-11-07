"""Build runs/lltDomainAdaptation/lltDomainAdaptation-T2-notebook.json from the Kaggle log of llt-da-notebook.
Usage: python notebook_runfile.py <kaggle .log (JSON list)> <kernel version> <out.json>
Cell times are differences between the log timestamps of each cell's printed result (seconds since kernel start)."""
import ast, json, sys
LOG, VER, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
L = json.load(open(LOG, encoding="utf-8"))
lines = []
for e in L:
    for ln in e["data"].splitlines():
        lines.append((e["time"], ln))

def first(sub):
    for t, ln in lines:
        if sub in ln:
            return t, ln
    raise KeyError(sub)

t0 = L[0]["time"]
t_end = first("{'closed0': (0.0022")[0]
panel_base = ast.literal_eval(first("{'hiBpb': 0.61307")[1].rsplit("}", 1)[0] + "}")
panel_cpt = ast.literal_eval(first("{'hiBpb': 0.61815")[1])
metrics = [json.loads(ln) for t, ln in lines if ln.startswith('{"step"')]
interp = []
for a in ("0.25", "0.5", "0.75"):
    ln = first(a + " {'hiBpb'")[1]
    interp.append({"a": float(a), **ast.literal_eval(ln.split(" ", 1)[1])})
qa_base = ast.literal_eval(first("{'closed0': (0.0, 0.0429)")[1].rsplit("}", 1)[0] + "}")
qa_ft = ast.literal_eval(first("{'closed0': (0.0022")[1].rsplit("}", 1)[0] + "}")
step0 = first("Hindi bpb, step 0")[1]
R = {
    "runId": "lltDomainAdaptation-T2-notebook", "module": "lltDomainAdaptation", "tier": "T2", "label": "baseline",
    "hardware": {"gpu": "Tesla T4", "count": 1, "vramGB": 15.6, "note": "Kaggle T4 x2 session, CUDA_VISIBLE_DEVICES=0"},
    "env": {"python": "3.13.16", "torch": "2.11.0+cu130", "cuda": "13.0", "transformersPreinstalled": "5.20.2",
            "transformers": "5.22.1", "peft": "0.21.2", "bitsandbytes": "0.50.2", "accelerate": "1.15.0", "datasets": "5.1.0",
            "tokenizers": "0.23.3", "huggingface_hub": "2.4.0", "sentence_transformers": "6.2.0", "gpu": "Tesla T4",
            "note": "python, torch and the preinstalled transformers are printed by the runtime-check cell; the other versions are the pins of the %pip -q install cell, which completed without error and prints nothing"},
    "notebookCommit": None,
    "config": {"notebook": "labs/lltDomainAdaptation/lltDomainAdaptation.ipynb", "kernel": "genieincodebottle/llt-da-notebook",
               "quickSteps": 150, "rowsPerStep": 8, "micro": 4, "peakLr": 3e-5, "replay": 0.05, "qloraEpochs": 2, "qloraBatch": 8, "qloraMicro": 2},
    "seed": 0,
    "metrics": metrics,
    "evals": {
        "panelBase": panel_base, "panelAfter150Steps": panel_cpt, "interpolation": interp,
        "gpuBeforeTraining": first("GPU before training")[1], "trainLine": first("peak GB 14.29")[1],
        "extensionStep0First100Texts": step0, "gpuBeforeQLoRA": first("GPU before QLoRA")[1],
        "qloraLine": first("181 steps |")[1], "qaBaseline": qa_base, "qaAfterQLoRA": qa_ft,
        "retrieval": [first("recall@" + k)[1] for k in ("1", "2", "5")],
        "fertility": [ln for t, ln in lines if "tokens per word" in ln or "tokens per Hindi word" in ln],
    },
    "cellSeconds": {
        "setupToPanelPrinted": round(first("{'hiBpb': 0.61307")[0] - t0, 1),
        "step3QaBaselines": float(first("1300 s")[1].rsplit("}", 1)[1].split()[0]),
        "step4CellStartToPanel": round(first("{'hiBpb': 0.61815")[0] - first("{'closed0': (0.0, 0.0429)")[0], 1),
        "step8Training": 1670.6,
        "step8Eval": float(first("596 s")[1].rsplit("}", 1)[1].split()[0]),
    },
    "notebookCellTimeSec": round(t_end - t0, 1),
    "firstCellToStep4PanelSec": round(first("{'hiBpb': 0.61815")[0] - t0, 1),
    "timeNote": "Seconds since kernel start from the Kaggle log timestamps. notebookCellTimeSec runs from the first log line to the last cell's printed result.",
    "samples": [{"q": "पीटरलू हत्याकांड के बाद किस कवि ने द मास्क ऑफ एनार्की लिखा था?", "gold": "पर्सी शेली", "pred": "पर्सी शेली"}],
    "wallClockSec": round(t_end - t0, 1), "costUSD": 0, "priceCheckedOn": None,
    "kaggleKernel": "genieincodebottle/llt-da-notebook", "kaggleKernelVersion": VER, "date": "2026-11-07",
}
json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(R["notebookCellTimeSec"], R["firstCellToStep4PanelSec"], R["cellSeconds"])
