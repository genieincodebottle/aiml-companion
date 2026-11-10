"""Stage 3a. On-policy samples: the SFT model writes k replies to every preference prompt.

On-policy means the replies come from the model you are about to train, so DPO compares replies it could
actually write, not replies from other, stronger models.
"""
from .common import TEMPLATE, Timer, cli, done, path, read_jsonl, stage_record, write_jsonl
from .gen import run_job


def main():
    a = cli("Stage 3a: on-policy samples", [(("--temperature",), dict(type=float, default=0.8))])
    cfg = a.cfg
    out = path(a, "pairs", "samples.jsonl")
    if done(out):
        print("stage 3a already done")
        return
    prompts = read_jsonl(path(a, "data", "pref_prompts.jsonl"))
    job = {"model": path(a, "sft", "merged", "x")[:-2], "backend": cfg["backend"], "n": cfg["k_samples"],
           "temperature": a.temperature, "top_p": 0.95, "max_new": cfg["sample_max_new"], "seed": a.seed,
           "dtype": cfg["dtype"], "max_model_len": 2048}
    with Timer() as t:
        res = run_job(job, [TEMPLATE.format(p["prompt"]) for p in prompts], path(a, "pairs", "samples_raw.json"))
    rows = [{"id": p["id"], "prompt": p["prompt"], "replies": r} for p, r in zip(prompts, res["results"])]
    write_jsonl(out, rows)
    flat = [o for r in rows for o in r["replies"]]
    stage_record(a, "sample", {
        "wallClockSec": t.sec, "genSec": res["wallClockSec"], "job": job, "n_prompts": len(rows), "n_replies": len(flat),
        "tokens": res["tokens"], "finishedShare": round(sum(o["finished"] for o in flat) / len(flat), 4),
        "meanTokens": round(sum(o["tokens"] for o in flat) / len(flat), 1),
        "examples": [{"prompt": rows[i]["prompt"][:400], "reply": rows[i]["replies"][0]["text"][:600]} for i in range(min(3, len(rows)))],
    })


if __name__ == "__main__":
    main()
