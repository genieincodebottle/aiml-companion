"""Build both extended tokenizers and measure fertility. Usage: python run_extend.py <data_dir> <work_dir> <out.json>"""
import json, sys, time, platform
import tokenizers, transformers
from transformers import AutoTokenizer
from extend_tokenizer import extend
D, W, OUT = sys.argv[1:4]
train = [json.loads(l)["text"] for i, l in zip(range(20000), open(f"{D}/hi_train.jsonl", encoding="utf-8"))]
hi = [json.loads(l)["text"] for l in open(f"{D}/hi_test.jsonl", encoding="utf-8")]
en = [json.loads(l)["text"] for l in open(f"{D}/en_edu.jsonl", encoding="utf-8")][:500]
hiW = sum(len(t.split()) for t in hi); enW = sum(len(t.split()) for t in en)
R = {"date": time.strftime("%Y-%m-%d"), "env": {"python": platform.python_version(), "tokenizers": tokenizers.__version__, "transformers": transformers.__version__},
     "trainDocs": "fineweb-2 hin_Deva train, first 20000 docs", "evalDocs": "hin_Deva test first 1000 docs; fineweb-edu first 500 docs", "rows": []}
base = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
def fert(tok, tag, extra):
    hiT = sum(len(x) for x in tok(hi, add_special_tokens=False)["input_ids"])
    enT = sum(len(x) for x in tok(en, add_special_tokens=False)["input_ids"])
    ok = all(tok.decode(tok(t, add_special_tokens=False)["input_ids"]) == t for t in hi[:200])
    # an English text must tokenize exactly as before
    same_en = all(tok(t, add_special_tokens=False)["input_ids"] == base(t, add_special_tokens=False)["input_ids"] for t in en)
    row = dict(tag=tag, lenTokenizer=len(tok), hiTokPerWord=round(hiT / hiW, 3), enTokPerWord=round(enT / enW, 3), hiTokens=hiT,
               roundTrip200=ok, englishIdsUnchanged500=same_en, eos=tok.eos_token_id, imEnd=tok.convert_tokens_to_ids("<|im_end|>"), id151643=tok.convert_ids_to_tokens(151643), **extra)
    R["rows"].append(row); print(row, flush=True)
fert(base, "qwen3-original", {})
for tag, mark, n, pin in [("ext10000", False, 10000, True), ("ext10000-markregex", True, 10000, True),
                          ("markregex-only", True, 0, True), ("broken-unpinned-ext10000", False, 10000, False)]:
    t = time.time()
    out, rep = extend("Qwen/Qwen3-0.6B-Base", train, n, mark, f"{W}/qwen3-{tag}", pin_special=pin)
    rep["buildSec"] = round(time.time() - t, 1)
    fert(AutoTokenizer.from_pretrained(out), tag, rep)
json.dump(R, open(OUT, "w"), indent=1)
