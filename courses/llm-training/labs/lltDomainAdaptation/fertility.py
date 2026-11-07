"""Fertility of three tokenizers on FineWeb2 Hindi (test split, first 1,000 docs) and FineWeb-Edu English (first 500 docs).
Usage: python fertility.py <data_dir> <out.json>"""
import json, sys, time, platform
import transformers, tokenizers
from transformers import AutoTokenizer
D, OUT = sys.argv[1], sys.argv[2]
hi = [json.loads(l)["text"] for l in open(f"{D}/hi_test.jsonl", encoding="utf-8")]
en = [json.loads(l)["text"] for l in open(f"{D}/en_edu.jsonl", encoding="utf-8")][:500]
R = {"date": time.strftime("%Y-%m-%d"), "env": {"python": platform.python_version(), "transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
     "data": {"hi": "HuggingFaceFW/fineweb-2 hin_Deva test, first 1000 docs", "en": "HuggingFaceFW/fineweb-edu sample-10BT train, first 500 docs"}, "rows": []}
for name, texts in [("hi", hi), ("en", en)]:
    R["data"][name + "Words"] = sum(len(t.split()) for t in texts)
    R["data"][name + "Bytes"] = sum(len(t.encode("utf-8")) for t in texts)
for mid in ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3.5-0.8B-Base", "HuggingFaceTB/SmolLM2-360M"]:
    tok = AutoTokenizer.from_pretrained(mid)
    row = {"model": mid, "lenTokenizer": len(tok)}
    for name, texts in [("hi", hi), ("en", en)]:
        n = sum(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])
        row[name + "Tokens"] = n
        row[name + "TokPerWord"] = round(n / R["data"][name + "Words"], 3)
        row[name + "BytesPerTok"] = round(R["data"][name + "Bytes"] / n, 3)
    row["hiOverEn"] = round(row["hiTokPerWord"] / row["enTokPerWord"], 2)
    R["rows"].append(row); print(row, flush=True)
json.dump(R, open(OUT, "w"), indent=1)
