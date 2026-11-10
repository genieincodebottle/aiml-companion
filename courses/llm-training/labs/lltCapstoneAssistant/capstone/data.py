"""Stage 1. Build the SFT set, the preference prompts and every evaluation set, decontaminate, and write the data card numbers.

Sources (licences checked on the Hugging Face API 2026-11-10):
  SFT        allenai/Dolci-Instruct-SFT (ODC-BY; per-source licences listed in its card; WildChat replies are GPT-4.1 outputs)
  prompts    HuggingFaceH4/ultrafeedback_binarized, prompts only (MIT)
  IFEval     google/IFEval (Apache-2.0), all 541 prompts
  win rate   lmarena-ai/arena-hard-auto, arena-hard-v2.0 questions (Apache-2.0)
  behaviour  XSTest (Rottger et al. 2024, CC-BY-4.0), the authors' GitHub release pinned to a commit
  judge check nvidia/HelpSteer3 preference (CC-BY-4.0): human pairwise labels
"""
import collections
import random

from .common import (BASE, IDENTITY, TEMPLATE, cli, done, mostly_english, ngrams, norm, path, read_jsonl, stage_record,
                     write_json, write_jsonl, Timer)

# Share of the SFT set per source group (sums to 1). Excluded on purpose: Aya (multilingual), Logic Puzzles,
# Tool Use (needs a tool format), Hardcoded Data (identity prompts that name another model), TableGPT, SciRiff.
MIX = {
    "chat": (0.37, ["Wildchat"]),
    "precise_if": (0.20, ["Dolci Instruct Precise IF"]),
    "flan": (0.05, ["FLAN"]),
    "oasst": (0.04, ["OpenAssistant"]),
    "math": (0.12, ["Tulu 3 Persona MATH", "Tulu 3 Persona GSM", "Tulu 3 Persona Algebra", "OpenMathInstruct 2"]),
    "code": (0.12, ["Evol CodeAlpaca", "Tulu 3 Persona Python", "Dolci Instruct Python Algorithms"]),
    "safety": (0.07, ["CoCoNot", "WildGuardMix", "WildJailbreak"]),
    "science": (0.03, ["Dolci Instruct OpenThoughts3+ Science"]),
}
XSTEST = "https://raw.githubusercontent.com/paul-rottger/xstest/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/xstest_prompts.csv"


def eval_sets(a, cfg, rng):
    import io
    import json
    import urllib.request
    import pandas as pd
    from datasets import load_dataset

    ife = load_dataset("google/IFEval", split="train")
    ifeval = [{"key": int(r["key"]), "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
               "kwargs": r["kwargs"]} for r in ife]
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("lmarena-ai/arena-hard-auto", "data/arena-hard-v2.0/question.jsonl", repo_type="dataset")
    with open(f, encoding="utf-8") as fh:            # split on newlines only: some prompts contain U+2028
        arena_all = [json.loads(line) for line in fh.read().split(chr(10)) if line.strip()]
    # Stratified: all creative-writing prompts are kept in proportion with the coding and maths ones
    by_sub = collections.defaultdict(list)
    for q in arena_all:
        by_sub[q.get("subcategory") or q["category"]].append(q)
    arena, subs = [], sorted(by_sub.items())
    for j, (sub, qs) in enumerate(subs):
        k = round(cfg["n_arena"] * len(qs) / len(arena_all)) if j < len(subs) - 1 else cfg["n_arena"] - len(arena)
        arena += [{"uid": q["uid"], "category": sub, "prompt": q["prompt"]} for q in rng.sample(qs, min(k, len(qs)))]
    xs = pd.read_csv(io.StringIO(urllib.request.urlopen(XSTEST, timeout=60).read().decode("utf-8")))
    xstest = [{"id": int(r.id), "prompt": r.prompt, "type": r.type, "label": r.label} for r in xs.itertuples()]
    hs = load_dataset("nvidia/HelpSteer3", "preference", split="validation")
    calib = []
    for r in hs:
        ctx = r["context"]
        if r["domain"] != "general" or r["language"] != "english" or len(ctx) != 1 or ctx[0]["role"] != "user":
            continue
        if abs(r["overall_preference"]) < 2 or len(r["response1"]) > 6000 or len(r["response2"]) > 6000:
            continue
        calib.append({"prompt": ctx[0]["content"], "a": r["response1"], "b": r["response2"],
                      "human": "A" if r["overall_preference"] < 0 else "B", "strength": abs(r["overall_preference"])})
    rng.shuffle(calib)
    calib = calib[:300 if not a.smoke else 8]
    return ifeval, arena, xstest, calib, arena_all


def main():
    a = cli("Stage 1: data")
    cfg, rng = a.cfg, random.Random(a.seed)
    out = {k: path(a, "data", k + ".jsonl") for k in ("sft", "sft_eval", "pref_prompts", "ifeval", "arena", "xstest", "calib")}
    if all(done(p) for p in out.values()) and done(path(a, "data", "card.json")):
        print("stage 1 already done")
        return
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(BASE)
    with Timer() as t:
        ifeval, arena, xstest, calib, arena_all = eval_sets(a, cfg, rng)
        if a.smoke:
            ifeval, xstest = ifeval[:8], xstest[:8]
        # Decontamination index: every 13-word span of every evaluation prompt, plus whole prompts for short ones
        grams, whole = set(), set()
        for q in ifeval + arena_all + xstest:
            grams |= ngrams(q["prompt"])
            whole.add(norm(q["prompt"]))

        def contaminated(text):
            return norm(text) in whole or bool(ngrams(text) & grams)

        # ------------------------------------------------ SFT pool
        n_hold = 300 if not a.smoke else 8                                     # held out for the eval loss
        want = {g: round(share * (cfg["n_sft"] + n_hold)) for g, (share, _) in MIX.items()}
        src2group = {s: g for g, (_, srcs) in MIX.items() for s in srcs}
        drops = collections.Counter()
        drops_by_group = collections.defaultdict(collections.Counter)
        pool = collections.defaultdict(list)
        shards = range(15) if not a.smoke else [0, 12]
        for i in shards:
            f = hf_hub_download("allenai/Dolci-Instruct-SFT", "data/train-%05d-of-00015.parquet" % i, repo_type="dataset")
            tbl = pq.read_table(f, columns=["id", "messages", "source_dataset"]).to_pylist()
            for r in tbl:
                g = src2group.get(r["source_dataset"])
                if g is None:
                    continue
                m = r["messages"]
                if len(m) != 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant":
                    drops["not single-turn"] += 1
                    continue
                if m[0].get("functions") or m[1].get("function_calls"):
                    drops["tool call"] += 1
                    continue
                pool[g].append((r["id"], r["source_dataset"], m[0]["content"], m[1]["content"]))
        seen, rows, per_src = set(), collections.defaultdict(list), collections.Counter()
        for g in MIX:
            cand = pool[g]
            rng.shuffle(cand)
            for rid, src, p, c in cand:
                if len(rows[g]) >= want[g]:
                    break
                if not p.strip() or not c.strip():
                    drops["empty"] += 1
                    drops_by_group[g]["empty"] += 1
                    continue
                if not (mostly_english(p) and mostly_english(c)):
                    drops["not English"] += 1
                    drops_by_group[g]["not English"] += 1
                    continue
                if IDENTITY.search(c):
                    drops["names a model or lab in the reply"] += 1
                    drops_by_group[g]["names a model or lab in the reply"] += 1
                    continue
                key = norm(p)
                if key in seen:
                    drops["duplicate prompt"] += 1
                    drops_by_group[g]["duplicate prompt"] += 1
                    continue
                if contaminated(p):
                    drops["overlaps an evaluation prompt"] += 1
                    drops_by_group[g]["overlaps an evaluation prompt"] += 1
                    continue
                n_p = len(tok(TEMPLATE.format(p)).input_ids)
                n_c = len(tok(c).input_ids) + 1
                if n_p > cfg["sft_max_len"] // 2 or n_p + n_c > cfg["sft_max_len"]:
                    drops["too long for max_length"] += 1
                    drops_by_group[g]["too long for max_length"] += 1
                    continue
                seen.add(key)
                rows[g].append({"id": rid, "group": g, "source": src, "prompt": TEMPLATE.format(p), "completion": c,
                                "raw_prompt": p, "n_prompt": n_p, "n_completion": n_c})
                per_src[src] += 1
        sft = [r for g in MIX for r in rows[g]]
        rng.shuffle(sft)
        sft_eval, sft = sft[:n_hold], sft[n_hold:]

        # ------------------------------------------------ preference prompts (only the prompts; replies come from our model)
        uf = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        pref_drops = collections.Counter()
        cand = list({norm(p): p for p in uf["prompt"]}.values())
        pref_drops["duplicate prompt"] = len(uf) - len(cand)
        rng.shuffle(cand)
        pref = []
        for p in cand:
            if len(pref) >= cfg["n_pref"]:
                break
            if not mostly_english(p):
                pref_drops["not English"] += 1
                continue
            if norm(p) in seen:
                pref_drops["also an SFT prompt"] += 1
                continue
            if contaminated(p):
                pref_drops["overlaps an evaluation prompt"] += 1
                continue
            if len(tok(TEMPLATE.format(p)).input_ids) > 512:
                pref_drops["prompt over 512 tokens"] += 1
                continue
            pref.append({"id": "uf-%06d" % len(pref), "prompt": p})

    for k, rows_ in (("sft", sft), ("sft_eval", sft_eval), ("pref_prompts", pref), ("ifeval", ifeval), ("arena", arena),
                     ("xstest", xstest), ("calib", calib)):
        write_jsonl(out[k], rows_)
    n_tok = sum(r["n_prompt"] + r["n_completion"] for r in sft)
    card = {
        "sft": {"n": len(sft), "n_heldout": len(sft_eval), "tokens": n_tok,
                "completion_tokens": sum(r["n_completion"] for r in sft),
                "per_group": dict(collections.Counter(r["group"] for r in sft)),
                "per_source": dict(per_src), "drops": dict(drops), "dropsByGroup": {g: dict(c) for g, c in drops_by_group.items()}, "max_len": cfg["sft_max_len"],
                "mean_completion_tokens": round(sum(r["n_completion"] for r in sft) / max(1, len(sft)), 1)},
        "pref": {"n": len(pref), "drops": dict(pref_drops), "pool": len(uf)},
        "eval": {"ifeval": len(ifeval), "arena": len(arena), "arena_per_category": dict(collections.Counter(q["category"] for q in arena)),
                 "xstest": len(xstest), "xstest_per_label": dict(collections.Counter(q["label"] for q in xstest)),
                 "calib": len(calib), "calib_human_A": sum(c["human"] == "A" for c in calib)},
        "decontamination": {"method": "13-word overlap or exact normalised match against all IFEval, Arena-Hard v2.0 and XSTest prompts",
                            "eval_ngrams": len(grams)},
    }
    write_json(path(a, "data", "card.json"), card)
    stage_record(a, "data", {"wallClockSec": t.sec, "card": card})


if __name__ == "__main__":
    main()
