import urllib.request
import pandas as pd
import ptd

HF = "https://huggingface.co/datasets/"
urllib.request.urlretrieve(HF + "allenai/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Easy/test/0000.parquet", "arc.parquet")
urllib.request.urlretrieve(HF + "Rowan/hellaswag/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet", "hs.parquet")
arc, hs = pd.read_parquet("arc.parquet"), pd.read_parquet("hs.parquet")
eval_texts = [q + " " + " ".join(c["text"]) for q, c in zip(arc["question"], arc["choices"])]
eval_texts += [c + " " + e[int(l)] for c, e, l in zip(hs["ctx"], hs["endings"], hs["label"])]
index = ptd.build_eval_index(eval_texts, n=13)
print("evaluation items", len(eval_texts), "| distinct 13-grams", len(index))

clean = [(u, t) for u, t in deduped if not ptd.contaminated(t, index)]
print("pages sharing a 13-gram with ARC-Easy or HellaSwag", len(deduped) - len(clean))

emails = ips = 0
scrubbed = []
for url, text in clean:
    text, n_email, n_ip = ptd.scrub_pii(text)
    emails += n_email
    ips += n_ip
    scrubbed.append((url, text))
print("emails replaced", emails, "| public IPv4 addresses replaced", ips)

question = arc.loc[arc["id"] == "Mercury_182158", "question"].iloc[0] + " wash hands"
paraphrase = "A pupil has just finished a lab exercise. Before walking out of the laboratory, what should the pupil do last? Wash their hands."
print("verbatim caught", ptd.contaminated("Practice question. " + question, index),
      "| paraphrase caught", ptd.contaminated("Practice question. " + paraphrase, index))
