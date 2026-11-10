"""Stage 5a. Generate the replies every evaluation scores: IFEval, the Arena-Hard sample and XSTest.

Models: base (our template, IFEval only, the starting point), sft, dpo, and the official instruct model
(Qwen3-1.7B in its own chat template with thinking switched off, so it answers directly as ours do).
Same decoding for every model: greedy for IFEval and XSTest, temperature 0.7 and top-p 0.8 for Arena-Hard
(Qwen's recommended non-thinking setting), and the same token cap.
"""
from .common import BASE, OFFICIAL, TEMPLATE, Timer, cli, path, read_jsonl, stage_record
from .gen import run_multi

SETS = {"ifeval": dict(temperature=0.0, top_p=1.0), "arena": dict(temperature=0.7, top_p=0.8),
        "xstest": dict(temperature=0.0, top_p=1.0, max_new=64)}


def main():
    a = cli("Stage 5a: evaluation generations", [(("--models",), dict(default="base,sft,dpo,official")),
                                                 (("--sets",), dict(default="ifeval,arena,xstest")),
                                                 (("--record",), dict(default="evalgen"))])
    cfg = a.cfg
    data = {s: read_jsonl(path(a, "data", s + ".jsonl")) for s in SETS}
    names = {"base": BASE, "official": OFFICIAL}
    rec = {}
    with Timer() as t:
        for m in a.models.split(","):
            model = names.get(m) or path(a, m, "merged", "x")[:-2]
            groups = []
            for s, dec in SETS.items():
                if (m == "base" and s != "ifeval") or s not in a.sets.split(","):
                    continue
                qs = data[s]
                if m == "official":
                    inputs = [[{"role": "user", "content": q["prompt"]}] for q in qs]
                    chat_kwargs = {"enable_thinking": False}
                else:
                    inputs = [TEMPLATE.format(q["prompt"]) for q in qs]
                    chat_kwargs = None
                groups.append({"inputs": inputs, "out": path(a, "eval", "gen_%s_%s.json" % (m, s)),
                               "params": {"n": 1, "temperature": dec["temperature"], "top_p": dec["top_p"],
                                          "max_new": dec.get("max_new", cfg["eval_max_new"]), "chat_kwargs": chat_kwargs}})
            margs = {"model": model, "backend": cfg["backend"], "seed": a.seed, "dtype": cfg["dtype"], "max_model_len": 16384,
                     "quant": None}
            for g, res in zip(groups, run_multi(margs, groups)):
                outs = [r[0] for r in res["results"]]
                key = "%s/%s" % (m, g["out"].rsplit("_", 1)[1][:-5])
                rec[key] = {"n": len(outs), "sec": res["wallClockSec"], "loadSec": res.get("loadSec"),
                            "finishedShare": round(sum(o["finished"] for o in outs) / len(outs), 4),
                            "meanTokens": round(sum(o["tokens"] for o in outs) / len(outs), 1)}
    stage_record(a, a.record, {"wallClockSec": t.sec, "sets": rec,
                                "decoding": {k: {**v, "max_new": v.get("max_new", cfg["eval_max_new"])} for k, v in SETS.items()}})


if __name__ == "__main__":
    main()
