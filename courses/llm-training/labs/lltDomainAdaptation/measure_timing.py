"""CPU timings for the lltDomainAdaptation page, for the serial quiet-machine pass. Prints one JSON line.
Usage: python measure_timing.py <data_dir>   (data_dir holds hi_train.jsonl, hi_test.jsonl from rag_prep/prep_local)
Keys:
  extend_build_min   build the 10,000-token Hindi extension of Qwen3's tokenizer from 20,000 docs (step 7)
  fertility_sec      tokenize 1,000 Hindi test docs with Qwen3 and with Qwen3.5 (step 2)
  rag_index_min      encode the 1,040-paragraph pool with multilingual-e5-small on CPU, 4 threads (step 3)
  bpb_100_hi_min     Hindi bits per byte of Qwen3-0.6B-Base on 100 held-out texts on CPU, 4 threads (step 7)
"""
import json, sys, time, tempfile
import torch
torch.set_num_threads(4)
from transformers import AutoTokenizer, AutoModelForCausalLM
import dalab as L
from extend_tokenizer import extend

D = sys.argv[1]
rd = lambda n: [json.loads(l) for l in open(D + "/" + n + ".jsonl", encoding="utf-8")]
hi_test = rd("hi_test")
hi_train = [json.loads(l)["text"] for i, l in zip(range(20000), open(D + "/hi_train.jsonl", encoding="utf-8"))]
out = {}
t = time.time()
for mid in ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3.5-0.8B-Base"):
    tok = AutoTokenizer.from_pretrained(mid)
    tok([d["text"] for d in hi_test], add_special_tokens=False)
out["fertility_sec"] = round(time.time() - t, 1)
t = time.time()
extend(L.MODEL, hi_train, 10000, False, tempfile.mkdtemp())
out["extend_build_min"] = round((time.time() - t) / 60, 1)
from sentence_transformers import SentenceTransformer
contexts, qa = L.load_xquad_hi()
pool = contexts + [x for x in (L.cut(d["text"], 1000) for d in hi_test[200:1000]) if x]
m = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
t = time.time()
m.encode(pool, prompt="passage: ", normalize_embeddings=True, batch_size=16)
out["rag_index_min"] = round((time.time() - t) / 60, 1)
tok = AutoTokenizer.from_pretrained(L.MODEL)
model = AutoModelForCausalLM.from_pretrained(L.MODEL, dtype=torch.float32).eval()
texts = L.eval_texts(hi_test[:400], 1000, 100, tok)
t = time.time()
L.bits_per_byte(model, tok, texts)
out["bpb_100_hi_min"] = round((time.time() - t) / 60, 1)
print(json.dumps(out))
