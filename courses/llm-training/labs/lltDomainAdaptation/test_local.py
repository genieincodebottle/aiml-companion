"""CPU smoke test of the evaluation code in dalab.py on tiny subsets (not a measurement)."""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import dalab as L

torch.set_num_threads(4)
D = sys.argv[1]
rd = lambda n: [json.loads(l) for l in open(D + "/" + n + ".jsonl", encoding="utf-8")]
hi_test, en_edu = rd("hi_test"), rd("en_edu")
tok = AutoTokenizer.from_pretrained(L.MODEL)
model = AutoModelForCausalLM.from_pretrained(L.MODEL, dtype=torch.float32).eval()
E = {"hi": L.eval_texts(hi_test[:400], 1000, 4, tok), "en": L.eval_texts(en_edu[:500], 3000, 4, tok),
     "xsc_hi": L.load_xstorycloze("hi")[:6], "xsc_en": L.load_xstorycloze("en")[:6], "hellaswag": L.load_hellaswag(6)}
t = time.time()
print(L.panel(model, tok, E), round(time.time() - t, 1))
split = json.load(open(sys.argv[2], encoding="utf-8"))
contexts, qa = L.load_xquad_hi()
byid = {x["id"]: x for x in qa}
distract = [x for x in (L.cut(d["text"], 1000) for d in hi_test[200:1000]) if x]
Q = {"test": [byid[i] for i in split["testIds"]][:3], "train": [byid[i] for i in split["trainIds"]], "shots": split["shots"],
     "pool": contexts + distract, "contexts": contexts, "retrieved": split["retrieved"]}
r = L.qa_eval(model, tok, Q, modes=("closed0", "rag0", "rag3"))
print(json.dumps(r, ensure_ascii=False)[:1500])
ex = L.qa_train_examples(tok, Q)
print(len(ex), len(ex[0][0]), sum(1 for v in ex[0][1] if v != -100), tok.decode([t for t, l in zip(*ex[0]) if l != -100]))
rows = L.pack_rows([d["text"] for d in en_edu[500:520]], tok, 128, 10)
mixed, n_en = L.mix_rows(rows, rows, 10, 0.25, 0)
print(rows.shape, n_en, [round(L.lr_at(s, 100, 1e-4, 4), 7) for s in (0, 3, 4, 50, 99)])
