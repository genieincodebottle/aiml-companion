"""Turn the Kaggle CPU kernel output (kaggle/out/llt-ptd-prep/ptd_prep_stats.json) into Run Library
files: lltPretrainData-cpu-funnel, -cpu-corpora and -cpu-classifier. Examples of removed pages are
kept as the URL, the rule and the first 200 characters only.

Usage: python record_prep.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from runlog import write_run  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "kaggle", "out", "llt-ptd-prep")
s = json.load(open(os.path.join(OUT, "ptd_prep_stats.json"), encoding="utf-8"))
# funnel_examples.json holds raw page excerpts, some from adult sites, so it stays out of this repo.
# Pass its path as the first argument.
ex = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "funnel_examples.json"), encoding="utf-8"))
kaggle = {"gpu": None, "count": 0, "vramGB": None, "cpu": "Kaggle CPU session, 4 cores", "kernel": "genieincodebottle/llt-ptd-prep v4"}
# Versions printed by the kernel's install cell (log llt-ptd-prep.log). warcio and datatrove printed no
# __version__; the kernel installed the PyPI latest on 2026-11-05, which PyPI lists as below.
env = {"python": "3.13.18 (Kaggle image)", "numpy": "2.1.3", "scikit-learn": "1.6.1", "tokenizers": "0.23.3", "pandas": "2.3.3",
       "trafilatura": "2.3.1", "datasketch": "2.0.0", "spacy": "3.8.16", "fasttext": "fasttext-numpy2 0.10.4",
       "warcio": "1.8.2 (PyPI latest on 2026-11-05)", "datatrove": "0.10.1 (PyPI latest on 2026-11-05)"}


# Curated examples only: the page quotes these, and none carries personal data. The raw example
# file stays in kaggle/out/ and is not published.
PICK = {
    "not_english": ("247wms.com", 160),
    "gopher_quality": ("beatrizcosials.com", 200),
    "gopher_repetition": ("memoryhacking.com", 160),
    "c4": ("biztechindia.com", 160),
}


def trim(v):
    out = {}
    for stage, (host, n) in PICK.items():
        for it in v[stage]:
            if host in it[0]:
                out[stage] = [str(x)[:n] for x in it[1:]]
                break
    out["near_duplicate_cluster"] = [str(x)[:120] for x in v["near_duplicate_clusters"][0][:1][0][1:]] + [v["near_duplicate_clusters"][0][-1]]
    return out


common = dict(extra={"hardware": kaggle, "env": env, "wallClockSec": s["timings"]["totalSec"]})
write_run("lltPretrainData-cpu-funnel", "T1", "baseline", {"source": s["source"], "decontam": s["decontam"]}, None, [],
          evals={k: s[k] for k in ["stages", "languagesTop", "ruleCounts", "datatroveAgreement", "funnelDedup", "datasketch",
                                   "pii", "funnelClassifier", "funnelWords", "wetComparison", "timings"]},
          notes={"examples": trim(ex)}, **common)
write_run("lltPretrainData-cpu-corpora", "T1", "baseline", {"c4": {k: s["c4"][k] for k in ("source", "licence", "docs", "bytes", "heldOutDocs")}},
          None, [], evals={"c4": s["c4"], "tokenizer": s["tokenizer"], "corpora": s["corpora"], "canaries": s["canaries"]}, **common)
write_run("lltPretrainData-cpu-classifier", "T1", "baseline", {"model": "HashingVectorizer(2**20, 1-2 grams) + TF-IDF + Ridge(alpha=1)"},
          0, [], evals=s["classifier"], **common)
