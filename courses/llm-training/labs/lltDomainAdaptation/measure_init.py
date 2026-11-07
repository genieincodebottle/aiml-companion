"""Step-0 effect of tokenizer extension and of the new-row initialization, on CPU (exact, no training).
Usage: python measure_init.py <data_dir> <tok_root> <out.json>"""
import json, sys, time, platform
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import dalab as L

torch.set_num_threads(4)
D, TR, OUT = sys.argv[1:4]
rd = lambda n: [json.loads(l) for l in open(D + "/" + n + ".jsonl", encoding="utf-8")]
hi_test, en_edu = rd("hi_test"), rd("en_edu")
base_tok = AutoTokenizer.from_pretrained(L.MODEL)
E = {"hi": L.eval_texts(hi_test[:400], 1000, 100, base_tok), "en": L.eval_texts(en_edu[:500], 3000, 100, base_tok)}
R = {"date": time.strftime("%Y-%m-%d"), "env": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__},
     "evalDocs": {"hi": len(E["hi"]), "en": len(E["en"]), "hiBytes": sum(len(t.encode()) for t in E["hi"]), "enBytes": sum(len(t.encode()) for t in E["en"])},
     "note": "Same 100 Hindi and 100 English held-out texts as the first 100 of the GPU panel. bits per byte, so tokenizers compare fairly.", "rows": []}
model0 = AutoModelForCausalLM.from_pretrained(L.MODEL, dtype=torch.float32).eval()
emb0 = model0.get_input_embeddings().weight.detach().clone()
R["embedding"] = {"rows": emb0.shape[0], "meanRowNorm_0_151643": round(float(emb0[:151643].norm(dim=1).mean()), 4),
                  "meanRowNorm_151669_151936": round(float(emb0[151669:].norm(dim=1).mean()), 4)}
print(R["embedding"], flush=True)


def run(tag, tok_dir, init):
    tok = AutoTokenizer.from_pretrained(tok_dir) if tok_dir else base_tok
    model = AutoModelForCausalLM.from_pretrained(L.MODEL, dtype=torch.float32).eval()
    if tok_dir and len(tok) > 151669:
        parts = json.load(open(tok_dir + "/new_token_parts.json"))
        if init == "mean":
            L.add_tokens_mean_init(model, len(tok), parts)
        else:
            model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)
    t = time.time()
    hi, en = L.bits_per_byte(model, tok, E["hi"]), L.bits_per_byte(model, tok, E["en"])
    # how often the model's top next-token guess on Hindi text is one of the new tokens
    row = {"tag": tag, "init": init, "vocabRows": model.get_input_embeddings().weight.shape[0], "lenTokenizer": len(tok),
           "hiBpb": round(hi["bpb"], 4), "hiTokens": hi["tokens"], "hiNllPerToken": round(hi["nllPerToken"], 4),
           "enBpb": round(en["bpb"], 4), "enTokens": en["tokens"], "sec": round(time.time() - t, 1)}
    R["rows"].append(row); print(row, flush=True)
    json.dump(R, open(OUT, "w"), indent=1)


run("original", None, None)
run("ext10000", TR + "/qwen3-ext10000", "mean")
run("ext10000", TR + "/qwen3-ext10000", "default")
run("broken-unpinned-ext10000", TR + "/qwen3-broken-unpinned-ext10000", "mean")
