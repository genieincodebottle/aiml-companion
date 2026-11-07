"""Build the question set and the RAG index, and measure retrieval recall on CPU.
Usage: python rag_prep.py <data_dir> <out_dir>
Writes <out_dir>/qa_split.json (question ids, retrieved paragraph ids) and the run file lltDomainAdaptation-t1-rag-index-v2026-11.json.
"""
import json, sys, time, platform, hashlib
import numpy as np
import torch, transformers, sentence_transformers
from sentence_transformers import SentenceTransformer
sys.path.insert(0, ".")
from dalab import load_xquad_hi, cut

torch.set_num_threads(4)
D, OUT = sys.argv[1], sys.argv[2]
contexts, qa = load_xquad_hi()
train = [x for x in qa if x["ctx"] < 140]
test = [x for x in qa if x["ctx"] >= 140]
hi_test = [json.loads(l) for l in open(D + "/hi_test.jsonl", encoding="utf-8")]
distract = [t for t in (cut(d["text"], 1000) for d in hi_test[200:1000]) if t]
pool = contexts + distract                     # ids 0-239 are the XQuAD paragraphs
pool_sha = hashlib.sha256("|".join(pool).encode()).hexdigest()[:16]
R = {"date": time.strftime("%Y-%m-%d"), "env": {"python": platform.python_version(), "torch": torch.__version__,
     "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__},
     "pool": {"xquadParagraphs": len(contexts), "distractors": len(distract), "total": len(pool), "sha": pool_sha},
     "questions": {"total": len(qa), "train": len(train), "test": len(test), "trainParagraphs": 140, "testParagraphs": 100},
     "retrievers": []}
retrieved = {}
for mid, qkw, pkw in [("intfloat/multilingual-e5-small", {"prompt": "query: "}, {"prompt": "passage: "}),
                      ]:
    m = SentenceTransformer(mid, device="cpu")
    t = time.time()
    P = m.encode(pool, batch_size=16, normalize_embeddings=True, **pkw)
    tp = time.time() - t
    Qe = m.encode([x["q"] for x in test], batch_size=32, normalize_embeddings=True, **qkw)
    top = np.argsort(-(Qe @ P.T), axis=1)[:, :5]
    rec = {k: float(np.mean([x["ctx"] in top[i, :k] for i, x in enumerate(test)])) for k in (1, 2, 5)}
    row = {"model": mid, "dim": int(P.shape[1]), "recallAt1": round(rec[1], 4), "recallAt2": round(rec[2], 4),
           "recallAt5": round(rec[5], 4), "encodePoolSec": round(tp, 1)}
    R["retrievers"].append(row); print(row, flush=True)
    if mid.startswith("intfloat"):
        retrieved = {x["id"]: [int(i) for i in top[i]] for i, x in enumerate(test)}
shots = [(x["q"], x["a"][0]) for x in train if len(x["a"][0]) <= 20][:3]
json.dump({"poolSha": pool_sha, "testIds": [x["id"] for x in test], "trainIds": [x["id"] for x in train],
           "shots": shots, "retrieved": retrieved}, open(OUT + "/qa_split.json", "w", encoding="utf-8"), ensure_ascii=False)
R["shots"] = shots
json.dump(R, open("C:/projects/aiml-companion/courses/llm-training/runs/lltDomainAdaptation/lltDomainAdaptation-t1-rag-index-v2026-11.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
