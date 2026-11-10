"""Turn the stage records into Run Library files (plan section 7.6 schema), one per stage, plus a short summary.

  python -m capstone.report --tier T3 --work <dir> --out <runs dir> [--price-usd-per-hour X --price-source URL --price-date D]
"""
import datetime
import glob
import json
import os

from .common import cli, path, read_json, write_json

MODULE = "lltCapstoneAssistant"


def main():
    a = cli("Run Library files", [(("--out",), dict(default=None)), (("--price-usd-per-hour",), dict(type=float, default=None)),
                                  (("--price-source",), dict(default=None)), (("--price-date",), dict(default=None)),
                                  (("--tag",), dict(default=""))])
    out = a.out or path(a, "runs", "x")[:-2]
    os.makedirs(out, exist_ok=True)
    recs = {os.path.basename(f)[:-5]: read_json(f) for f in glob.glob(path(a, "records", "*.json"))}
    summary = {}
    for stage, r in sorted(recs.items()):
        run_id = "%s-%s-%s%s%s" % (MODULE, a.tier, stage, "-smoke" if r.get("smoke") else "", a.tag)
        sec = r.get("wallClockSec")
        gpu_stage = stage not in ("data", "scores")
        cost = round(sec / 3600 * a.price_usd_per_hour, 2) if (sec and a.price_usd_per_hour and gpu_stage) else None
        body = {k: v for k, v in r.items() if k not in ("stage", "tier", "smoke", "seed", "date", "hardware", "env", "command",
                                                        "wallClockSec", "config")}
        rec = {"runId": run_id, "module": MODULE, "tier": a.tier, "label": "smoke" if r.get("smoke") else "reference:" + stage,
               "hardware": r.get("hardware"), "env": r.get("env"), "notebookCommit": None, "config": r.get("config", {}),
               "seed": r.get("seed"), "metrics": body.pop("trainLoss", None) or body.pop("train", None) or [],
               "evals": body, "samples": body.pop("examples", []), "wallClockSec": sec, "costUSD": cost,
               "price": {"usdPerHour": a.price_usd_per_hour, "source": a.price_source} if cost is not None else None,
               "priceCheckedOn": a.price_date if cost is not None else None, "date": r.get("date", "")[:10],
               "command": r.get("command")}
        write_json(os.path.join(out, run_id + ".json"), rec)
        summary[stage] = {"runId": run_id, "wallClockSec": sec, "gpu": gpu_stage}
    gpu_sec = sum(v["wallClockSec"] or 0 for v in summary.values() if v["gpu"])
    write_json(os.path.join(out, "%s-%s-summary%s.json" % (MODULE, a.tier, a.tag)),
               {"stages": summary, "gpuStageSec": round(gpu_sec, 1), "gpuStageHours": round(gpu_sec / 3600, 2),
                "written": datetime.date.today().isoformat()})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
