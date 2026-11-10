"""Probe before training: prompt style (zero-shot vs one-shot) and precision (fp16 vs bf16) on the base models.
Greedy reply + 8 samples at temperature 1.0 on 100 test problems (50 with 3 numbers, 50 with 4)."""
import json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import countdown as cd, passk, rl_core

out = {}
test = [json.loads(l) for l in open(HERE / "data" / "test.jsonl")]
test = test[:50] + test[100:150]
golds = [cd.problem_text(p) for p in test]
N = 8
for model_id in sys.argv[2:]:
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    ns = rl_core.load(model_id, 512, cd.STOPS, tok)
    for dtype in ["fp16", "bf16"]:
        dt = torch.float16 if dtype == "fp16" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt).cuda().eval()
        ns["autocast"] = lambda dt=dt: torch.autocast("cuda", dtype=dt)
        for style in ["zero", "oneshot"]:
            cd.PROMPT_STYLE = style
            prompts = [cd.prompt(g) for g in golds]
            t = time.time()
            g_rows = ns["sample"](model, prompts, temperature=0, batch=100)
            s_rows = ns["sample"](model, [p for p in prompts for _ in range(N)], temperature=1.0, batch=200)
            sec = time.time() - t
            dec = lambda rows: [tok.decode(r) for r in rows]
            g_rep, s_rep = dec(g_rows), dec(s_rows)
            g_ok = [cd.correct(r, g) for r, g in zip(g_rep, golds)]
            counts = [sum(cd.correct(s_rep[i * N + j], g) for j in range(N)) for i, g in enumerate(golds)]
            nonascii = sum(sum(ord(c) > 127 for c in r) / max(1, len(r)) for r in s_rep) / len(s_rep)
            key = model_id.split("/")[-1] + "|" + dtype + "|" + style
            out[key] = dict(greedy=sum(g_ok) / len(g_ok), greedy3=sum(g_ok[:50]) / 50, greedy4=sum(g_ok[50:]) / 50,
                            passAtK=passk.curve(counts, N, [1, 2, 4, 8]), tagRate=sum(cd.extract_answer(r) is not None for r in s_rep) / len(s_rep),
                            meanTokens=sum(map(len, s_rows)) / len(s_rows), greedyTokens=sum(map(len, g_rows)) / len(g_rows),
                            nonAsciiShare=round(nonascii, 4), sec=round(sec, 1), greedyExample=g_rep[0][:500], sampleExamples=[s_rep[k][:400] for k in (0, 1, 8)])
            print(key, json.dumps({k: v for k, v in out[key].items() if "Example" not in k}), flush=True)
            json.dump(out, open(sys.argv[1], "w"), indent=1)
        del model; torch.cuda.empty_cache()
