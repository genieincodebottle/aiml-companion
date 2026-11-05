"""lltPretrainData data prep. Runs on a Kaggle CPU kernel (llt-ptd-prep, 4 cores, no GPU quota).

Part A, the funnel. One Common Crawl WARC file (CC-MAIN-2024-10, segment file 00000) through
extraction, language ID, Gopher quality, Gopher repetition, C4 rules, exact and MinHash dedup,
benchmark decontamination and the PII scrub, with the document count after every stage and
removed examples. Cross-checks: datatrove's filters on the same documents, datasketch's MinHash LSH
on the same documents, and the matching WET file's text for the same URLs.

Part B, the quality classifier. A hashed bag-of-words ridge regression trained on the FineWeb-Edu
Llama 3 annotations (one parquet shard), scored on a held-out split.

Part C, the training corpora. C4 en.noclean shard 00000 (CC WET text with a language filter only)
as the unfiltered corpus U, the same documents through our filters as F, the top half of F by
classifier score as E, plus wikitext-103 and TinyStories for the mixture runs. Tokenized with one
4,096-token byte-level BPE trained on a sample of all three domains. Also writes the canaries, the
held-out evaluation sets and the train/val leakage split for the GPU kernel.

Usage: python prep_ptd.py <out_dir> [--small]
"""
import gzip, hashlib, io, json, os, random, re, signal, sys, time, urllib.request
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ptd  # noqa: E402

OUT = sys.argv[1]
SMALL = "--small" in sys.argv
RAW = os.environ.get("PTD_RAW", os.path.join(OUT, "raw"))
os.makedirs(OUT, exist_ok=True); os.makedirs(RAW, exist_ok=True)
T0 = time.time()
STATS = {"small": SMALL, "stages": {}, "timings": {}}
CC = "https://data.commoncrawl.org/"
WARC = "crawl-data/CC-MAIN-2024-10/segments/1707947473347.0/warc/CC-MAIN-20240220211055-20240221001055-00000.warc.gz"
WET = WARC.replace("/warc/", "/wet/").replace(".warc.gz", ".warc.wet.gz")
HF = "https://huggingface.co/datasets/"
C4_NOCLEAN = HF + "allenai/c4/resolve/main/en.noclean/c4-train.00000-of-07168.json.gz"
EDU = HF + "HuggingFaceFW/fineweb-edu-llama3-annotations/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
ARC = HF + "allenai/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Easy/test/0000.parquet"
HSW = HF + "Rowan/hellaswag/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
WIKI_TR = HF + "Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/train/0000.parquet"
WIKI_TE = HF + "Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/test/0000.parquet"
TS_TR = HF + "roneneldan/TinyStories/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
TS_VA = HF + "roneneldan/TinyStories/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
LID = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def log(*a):
    print(round(time.time() - T0, 1), *a, flush=True)


def save():
    STATS["timings"]["totalSec"] = round(time.time() - T0, 1)
    with open(os.path.join(OUT, "ptd_prep_stats.json"), "w") as f:
        json.dump(STATS, f, indent=1)


def dl(url, name, max_bytes=None, tries=10):
    path = os.path.join(RAW, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llt-ptd-prep/1.0"})
            if max_bytes:
                req.add_header("Range", "bytes=0-%d" % (max_bytes - 1))
            with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(path + ".part", path)
            log("downloaded", name, os.path.getsize(path))
            return path
        except Exception as e:  # Common Crawl answers SlowDown under load; back off and retry
            log("download retry", t, name, repr(e)[:120])
            time.sleep(min(300, 15 * 2 ** t))
    raise RuntimeError("download failed " + url)


# ---------------------------------------------------------------------------------------------
# Workers (module level so multiprocessing can pickle them)
_LID = None


def _alarm(signum, frame):
    raise TimeoutError()


def w_extract(item):
    url, html = item
    import trafilatura
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(5)
    try:
        txt = trafilatura.extract(html, favor_precision=True, include_comments=False, deduplicate=True)
    except Exception:
        txt = None
    finally:
        if has_alarm:
            signal.alarm(0)
    return url, txt or ""


def w_lang(text):
    global _LID
    if _LID is None:
        import fasttext
        fasttext.FastText.eprint = lambda x: None
        _LID = fasttext.load_model(os.path.join(RAW, "lid.176.bin"))
    # The low-level call: fasttext's predict() wrapper uses np.array(copy=False), which NumPy 2 rejects.
    t = text.replace("\n", " ")[:20000]
    try:
        prob, label = _LID.f.predict(t, 1, 0.0, "strict")[0]
    except Exception:  # builds whose low-level binding differs (fasttext-numpy2)
        labels, probs = _LID.predict(t, k=1)
        label, prob = labels[0], probs[0]
    return label.replace("__label__", ""), float(prob)


def w_filters(text):
    """Gopher quality, Gopher repetition, C4. Returns (stage_failed, rule, cleaned_text)."""
    r = ptd.gopher_quality(text)
    if r:
        return "gopher_quality", r, None
    r = ptd.gopher_repetition(text)
    if r:
        return "gopher_repetition", r, None
    cleaned, r = ptd.c4_clean(text)
    if r:
        return "c4", r, None
    return None, None, cleaned


def w_c4_noterm(text):
    _, r = ptd.c4_clean(text, terminal_punct=False)
    return r


A_HASH, B_HASH = ptd.make_hashes(112, seed=1)


def w_sig(text):
    return ptd.minhash(ptd.shingles(text), A_HASH, B_HASH)


def w_datatrove(text):
    try:
        from datatrove.data import Document
        from datatrove.pipeline.filters import GopherQualityFilter, GopherRepetitionFilter, C4QualityFilter
        global _DT
        try:
            _DT
        except NameError:
            # C4 with the paper's numbers (3 sentences per page, 5 words per line), as ptd.c4_clean uses
            _DT = (GopherQualityFilter(), GopherRepetitionFilter(), C4QualityFilter(min_num_sentences=3, min_words_per_line=5))
        out = []
        for f in _DT:
            d = Document(text=text, id="x")
            r = f.filter(d)
            out.append(r is True)
        return out
    except Exception as e:
        return repr(e)[:200]


def w_datasketch(text):
    from datasketch import MinHash
    m = MinHash(num_perm=112, seed=1)
    w = ptd.norm_text(text).split()
    for g in {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}:
        m.update(g.encode())
    return m


def dedup(texts, pool):
    """Exact dedup, then MinHash LSH (14 bands x 8 rows) over the survivors. Returns kept indices
    and stats, including the exact Jaccard of every candidate pair."""
    seen, keep_exact = {}, []
    for i, t in enumerate(texts):
        k = ptd.exact_key(t)
        if k not in seen:
            seen[k] = i; keep_exact.append(i)
    sigs = np.stack(pool.map(w_sig, [texts[i] for i in keep_exact], chunksize=64))
    pairs = ptd.lsh_candidates(sigs, 14, 8)
    roots = ptd.clusters(len(keep_exact), pairs)
    keep = [keep_exact[i] for i in range(len(keep_exact)) if roots[i] == i]
    sample = sorted(pairs)[:3000]
    js = [ptd.jaccard(ptd.shingles(texts[keep_exact[a]]), ptd.shingles(texts[keep_exact[b]])) for a, b in sample]
    est = [float((sigs[a] == sigs[b]).mean()) for a, b in sample]
    st = {"in": len(texts), "exactDuplicatesRemoved": len(texts) - len(keep_exact), "candidatePairs": len(pairs),
          "nearDuplicatesRemoved": len(keep_exact) - len(keep), "out": len(keep),
          "pairsChecked": len(sample), "pairJaccardMean": round(float(np.mean(js)), 4) if js else None,
          "pairJaccardBelow0.5": int(sum(j < 0.5 for j in js)), "pairJaccardAtLeast0.8": int(sum(j >= 0.8 for j in js)),
          "estVsExactMeanAbsErr": round(float(np.mean(np.abs(np.array(js) - np.array(est)))), 4) if js else None}
    return keep, st, sigs, keep_exact, pairs


def short(t, n=400):
    return re.sub(r"\s+", " ", t)[:n]


# ---------------------------------------------------------------------------------------------
def eval_sets():
    import pandas as pd
    arc = pd.read_parquet(dl(ARC, "arc_easy_test.parquet"))
    hs = pd.read_parquet(dl(HSW, "hellaswag_val.parquet"))
    arc_texts = [q + " " + " ".join(c["text"]) for q, c in zip(arc["question"], arc["choices"])]
    hs_texts = [c + " " + e[int(l)] for c, e, l in zip(hs["ctx"], hs["endings"], hs["label"])]
    STATS["decontam"] = {"arcEasyTestItems": len(arc_texts), "hellaswagValItems": len(hs_texts), "n": 13}
    idx = ptd.build_eval_index(arc_texts + hs_texts, 13)
    STATS["decontam"]["indexNgrams"] = len(idx)
    return idx, hs


def part_a(pool, decon, score=None):
    from warcio.archiveiterator import ArchiveIterator
    t = time.time()
    path = dl(CC + WARC, "warc.gz", max_bytes=(30 << 20) if SMALL else None)
    n_rec, n_resp, items = 0, 0, []
    try:
        with open(path, "rb") as f:
            for rec in ArchiveIterator(f):
                n_rec += 1
                if rec.rec_type != "response":
                    continue
                n_resp += 1
                ctype = (rec.http_headers.get_header("Content-Type") or "") if rec.http_headers else ""
                status = rec.http_headers.get_statuscode() if rec.http_headers else ""
                if status != "200" or "html" not in ctype.lower():
                    continue
                body = rec.content_stream().read()
                if len(body) > 2_000_000:
                    continue
                items.append((rec.rec_headers.get_header("WARC-Target-URI"), body.decode("utf-8", "replace")))
    except Exception as e:  # a range download ends mid-record
        log("warc read stopped", repr(e)[:100])
    S = STATS["stages"]
    S["warcRecords"] = n_rec; S["httpResponses"] = n_resp; S["html200"] = len(items)
    STATS["source"] = {"warc": CC + WARC, "warcBytes": os.path.getsize(path), "wet": CC + WET,
                       "licence": "Common Crawl Terms of Use (commoncrawl.org/terms-of-use); page text is copyright of each site"}
    log("warc", n_rec, n_resp, len(items))
    te = time.time()
    ext = pool.map(w_extract, items, chunksize=16)
    STATS["timings"]["extractSec"] = round(time.time() - te, 1)
    del items
    docs = [(u, x) for u, x in ext if x.strip()]
    S["extractedNonEmpty"] = len(docs)
    ex = {"extraction_empty": [u for u, x in ext if not x.strip()][:20]}
    langs = pool.map(w_lang, [x for _, x in docs], chunksize=64)
    lang_count = {}
    for l, p in langs:
        lang_count[l] = lang_count.get(l, 0) + 1
    STATS["languagesTop"] = sorted(lang_count.items(), key=lambda kv: -kv[1])[:15]
    en = [(u, x) for (u, x), (l, p) in zip(docs, langs) if l == "en" and p >= 0.65]
    ex["not_english"] = [(u, l, round(p, 2), short(x, 160)) for (u, x), (l, p) in zip(docs, langs) if not (l == "en" and p >= 0.65)][:20]
    S["english065"] = len(en)
    res = pool.map(w_filters, [x for _, x in en], chunksize=32)
    rules = {}
    kept = []
    for (u, x), (stage, rule, cleaned) in zip(en, res):
        if stage:
            rules.setdefault(stage, {}); rules[stage][rule] = rules[stage].get(rule, 0) + 1
            ex.setdefault(stage, [])
            if len(ex[stage]) < 40:
                ex[stage].append((u, rule, short(x)))
        else:
            kept.append((u, cleaned))
    STATS["ruleCounts"] = rules
    S["afterGopherQuality"] = len(en) - sum(rules.get("gopher_quality", {}).values())
    S["afterGopherRepetition"] = S["afterGopherQuality"] - sum(rules.get("gopher_repetition", {}).values())
    S["afterC4"] = len(kept)
    # C4 without the terminal-punctuation line rule, on the documents that reached the C4 stage
    reach = [x for (u, x), (stage, _, _) in zip(en, res) if stage in (None, "c4")]
    noterm = pool.map(w_c4_noterm, reach, chunksize=32)
    S["afterC4_noTerminalPunctRule"] = S["afterGopherRepetition"] - sum(r is not None for r in noterm)
    # Cross-check against datatrove's implementation of the same three filters
    dt = pool.map(w_datatrove, [x for _, x in en], chunksize=32)
    if dt and isinstance(dt[0], list):
        ours_q = [ptd.gopher_quality(x) is None for _, x in en]
        ours_r = [ptd.gopher_repetition(x) is None for _, x in en]
        ours_c = [ptd.c4_clean(x)[1] is None for _, x in en]
        STATS["datatroveAgreement"] = {"docs": len(en),
                                       "gopherQuality": round(float(np.mean([a == d[0] for a, d in zip(ours_q, dt)])), 4),
                                       "gopherRepetition": round(float(np.mean([a == d[1] for a, d in zip(ours_r, dt)])), 4),
                                       "c4": round(float(np.mean([a == d[2] for a, d in zip(ours_c, dt)])), 4),
                                       "datatroveKeepsAll3": int(sum(all(d) for d in dt)), "oursKeepsAll3": len(kept)}
    else:
        STATS["datatroveAgreement"] = {"error": str(dt[0]) if dt else "no docs"}
    texts = [x for _, x in kept]
    keep, dst, sigs, keep_exact, pairs = dedup(texts, pool)
    STATS["funnelDedup"] = dst
    S["afterExactDedup"] = dst["in"] - dst["exactDuplicatesRemoved"]
    S["afterMinHashDedup"] = dst["out"]
    roots = ptd.clusters(len(keep_exact), pairs)
    groups = {}
    for i, r in enumerate(roots):
        groups.setdefault(r, []).append(keep_exact[i])
    big = sorted(groups.values(), key=len, reverse=True)[:5]
    ex["near_duplicate_clusters"] = [[(kept[i][0], short(kept[i][1], 200)) for i in g[:3]] + [len(g)] for g in big]
    # datasketch on the same exact-deduplicated documents, same 14 x 8 banding
    try:
        from datasketch import MinHashLSH
        ms = pool.map(w_datasketch, [texts[i] for i in keep_exact], chunksize=64)
        lsh = MinHashLSH(num_perm=112, params=(14, 8))
        dpairs = set()
        for i, m in enumerate(ms):
            for j in lsh.query(m):
                dpairs.add((min(i, int(j)), max(i, int(j))))
            lsh.insert(str(i), m)
        droots = ptd.clusters(len(ms), dpairs)
        STATS["datasketch"] = {"candidatePairs": len(dpairs), "nearDuplicatesRemoved": len(ms) - sum(1 for i, r in enumerate(droots) if r == i),
                               "pairsInBoth": len(dpairs & pairs)}
    except Exception as e:
        STATS["datasketch"] = {"error": repr(e)[:200]}
    final = [kept[i] for i in sorted(keep)]
    dec = [ptd.contaminated(x, decon) for _, x in final]
    ex["contaminated"] = [(u, short(x, 300)) for (u, x), c in zip(final, dec) if c][:20]
    final = [d for d, c in zip(final, dec) if not c]
    S["afterDecontamination"] = len(final)
    n_e = n_i = n_docs = 0
    scrubbed = []
    for u, x in final:
        y, a, b = ptd.scrub_pii(x)
        n_e += a; n_i += b; n_docs += (a + b) > 0
        scrubbed.append((u, y))
    STATS["pii"] = {"emailsReplaced": n_e, "ipsReplaced": n_i, "docsChanged": n_docs, "docs": len(scrubbed)}
    S["final"] = len(scrubbed)
    if score is not None and scrubbed:
        sc = score([x for _, x in scrubbed])
        cut = STATS["classifier"]["tunedCut"]
        STATS["funnelClassifier"] = {"atLeast3": int((sc >= 3).sum()), "atLeastTunedCut": int((sc >= cut).sum()),
                                     "tunedCut": cut, "median": round(float(np.median(sc)), 3)}
        order = np.argsort(sc)
        ex["classifier_low"] = [(scrubbed[i][0], round(float(sc[i]), 2), short(scrubbed[i][1], 300)) for i in order[:15]]
        ex["classifier_high"] = [(scrubbed[i][0], round(float(sc[i]), 2), short(scrubbed[i][1], 300)) for i in order[::-1][:15]]
    STATS["funnelWords"] = {"final": int(sum(len(x.split()) for _, x in scrubbed)),
                            "extracted": int(sum(len(x.split()) for _, x in docs))}
    with open(os.path.join(OUT, "funnel_examples.json"), "w") as f:
        json.dump(ex, f, indent=1)
    with open(os.path.join(OUT, "funnel_final_docs.jsonl"), "w") as f:
        for u, x in scrubbed:
            f.write(json.dumps({"url": u, "text": x}) + "\n")
    STATS["timings"]["funnelSec"] = round(time.time() - t, 1)
    save()
    # The matching WET file: Common Crawl's own text extraction of the same pages
    try:
        from warcio.archiveiterator import ArchiveIterator
        wpath = dl(CC + WET, "wet.gz", max_bytes=(10 << 20) if SMALL else None, tries=6)
        wet = {}
        try:
            with open(wpath, "rb") as f:
                for rec in ArchiveIterator(f):
                    if rec.rec_type == "conversion":
                        wet[rec.rec_headers.get_header("WARC-Target-URI")] = len(rec.content_stream().read().decode("utf-8", "replace").split())
        except Exception as e:
            log("wet read stopped", repr(e)[:100])
        tw = {u: len(x.split()) for u, x in docs}
        both = [u for u in tw if u in wet]
        STATS["wetComparison"] = {"wetRecords": len(wet), "urlsInBoth": len(both),
                                  "medianWordsTrafilatura": float(np.median([tw[u] for u in both])) if both else None,
                                  "medianWordsWET": float(np.median([wet[u] for u in both])) if both else None}
    except Exception as e:
        STATS["wetComparison"] = {"error": repr(e)[:200]}
    save()
    return scrubbed


def part_b():
    import pandas as pd
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    t = time.time()
    df = pd.read_parquet(dl(EDU, "edu_ann_0000.parquet"), columns=["text", "score"])
    if SMALL:
        df = df.iloc[:6000]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(df))
    n_test = len(df) // 6
    te, tr = perm[:n_test], perm[n_test:]
    hv = HashingVectorizer(n_features=2 ** 20, alternate_sign=False, ngram_range=(1, 2), norm=None)
    tf = TfidfTransformer(sublinear_tf=True)
    Xtr = tf.fit_transform(hv.transform(df["text"].iloc[tr]))
    Xte = tf.transform(hv.transform(df["text"].iloc[te]))
    ytr, yte = df["score"].values[tr], df["score"].values[te]
    reg = Ridge(alpha=1.0).fit(Xtr, ytr)
    p = reg.predict(Xte)
    # A ridge regression shrinks toward the mean score, so a cut at 3.0 keeps almost nothing. Pick the
    # cut on a slice of the TRAIN split (never on test), then report test F1 at that cut and AUC.
    from sklearn.metrics import roc_auc_score
    n_tune = len(tr) // 5
    reg_t = Ridge(alpha=1.0).fit(Xtr[n_tune:], ytr[n_tune:])
    pt = reg_t.predict(Xtr[:n_tune])
    cuts = np.arange(1.5, 3.01, 0.05)
    f1s = [f1_score(ytr[:n_tune] >= 3, pt >= c) for c in cuts]
    best_cut = float(cuts[int(np.argmax(f1s))])
    STATS["classifier"] = {"data": EDU, "rows": len(df), "train": len(tr), "test": len(te),
                           "scoreCounts": {int(k): int(v) for k, v in df["score"].value_counts().sort_index().items()},
                           "testMSE": round(float(np.mean((p - yte) ** 2)), 4),
                           "testMSE_predictMean": round(float(np.mean((ytr.mean() - yte) ** 2)), 4),
                           "f1At3": round(float(f1_score(yte >= 3, p >= 3)), 4),
                           "precisionAt3": round(float(precision_score(yte >= 3, p >= 3, zero_division=0)), 4),
                           "recallAt3": round(float(recall_score(yte >= 3, p >= 3)), 4),
                           "confusionAt3": confusion_matrix(yte >= 3, p >= 3).tolist(),
                           "positiveRateTest": round(float(np.mean(yte >= 3)), 4),
                           "aucAt3": round(float(roc_auc_score(yte >= 3, p)), 4),
                           "tunedCut": round(best_cut, 2),
                           "f1AtTunedCut": round(float(f1_score(yte >= 3, p >= best_cut)), 4),
                           "precisionAtTunedCut": round(float(precision_score(yte >= 3, p >= best_cut, zero_division=0)), 4),
                           "recallAtTunedCut": round(float(recall_score(yte >= 3, p >= best_cut)), 4),
                           "trainSec": round(time.time() - t, 1)}
    log("classifier", STATS["classifier"])
    save()

    def score(texts):
        return reg.predict(tf.transform(hv.transform(texts)))
    return score


def doc_tokens(tok, texts, eot):
    out = []
    for i in range(0, len(texts), 2000):
        for e in tok.encode_batch(texts[i:i + 2000]):
            out.append(np.asarray(e.ids + [eot], dtype=np.uint16))
    return out


def save_corpus(name, docs, cap=None):
    lens = np.array([d.size for d in docs], dtype=np.int64)
    if cap is not None:
        n = int(np.searchsorted(np.cumsum(lens), cap)) + 1
        docs, lens = docs[:n], lens[:n]
    arr = np.concatenate(docs) if docs else np.zeros(0, np.uint16)
    np.save(os.path.join(OUT, name + ".npy"), arr)
    np.save(os.path.join(OUT, name + "_offsets.npy"), np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64))
    STATS.setdefault("corpora", {})[name] = {"docs": len(docs), "tokens": int(arr.size),
                                             "meanDocTokens": round(float(lens.mean()), 1) if len(lens) else 0}
    log("saved", name, len(docs), arr.size)


def part_c(pool, decon, score):
    import pandas as pd
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    t = time.time()
    path = dl(C4_NOCLEAN, "c4noclean_00000.json.gz")
    raw = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line)["text"])
            if SMALL and len(raw) >= 4000:
                break
    rng = random.Random(0)
    rng.shuffle(raw)
    n_hold = 400 if SMALL else 6000
    hold, utrain = raw[:n_hold], raw[n_hold:]
    C = STATS["c4"] = {"source": C4_NOCLEAN, "licence": "ODC-By 1.0 (allenai/c4), plus Common Crawl Terms of Use",
                       "docs": len(raw), "bytes": int(sum(len(x.encode()) for x in raw)), "heldOutDocs": n_hold}
    # Leakage between U's train and held-out split: MinHash over both, held-out docs with a candidate in train
    sig_tr = np.stack(pool.map(w_sig, utrain, chunksize=64))
    sig_ho = np.stack(pool.map(w_sig, hold, chunksize=64))
    allsig = np.concatenate([sig_ho, sig_tr])
    pairs = ptd.lsh_candidates(allsig, 14, 8)
    leak = set()
    for a, b in pairs:
        if a < n_hold <= b and float((allsig[a] == allsig[b]).mean()) >= 0.8:
            leak.add(a)
    exact_tr = {ptd.exact_key(x) for x in utrain}
    exact_leak = {i for i, x in enumerate(hold) if ptd.exact_key(x) in exact_tr}
    C["leak"] = {"heldOut": n_hold, "nearDupInTrain_est0.8": len(leak), "exactDupInTrain": len(exact_leak),
                 "leakedAny": len(leak | exact_leak)}
    leak_idx = sorted(leak | exact_leak)
    # Filtered corpus F: same stages as the funnel, from language ID on
    langs = pool.map(w_lang, raw, chunksize=64)
    en = [i for i, (l, p) in enumerate(langs) if l == "en" and p >= 0.65]
    C["english065"] = len(en)
    res = pool.map(w_filters, [raw[i] for i in en], chunksize=32)
    rules = {}
    passed, cleaned = [], {}
    for i, (stage, rule, c) in zip(en, res):
        if stage:
            rules.setdefault(stage, {}); rules[stage][rule] = rules[stage].get(rule, 0) + 1
        else:
            passed.append(i); cleaned[i] = c
    C["ruleCounts"] = rules; C["afterFilters"] = len(passed)
    texts = [cleaned[i] for i in passed]
    keep, dst, _, _, _ = dedup(texts, pool)
    C["dedup"] = dst
    fid = [passed[k] for k in sorted(keep)]
    dec = [ptd.contaminated(cleaned[i], decon) for i in fid]
    C["contaminatedRemoved"] = int(sum(dec))
    fid = [i for i, d in zip(fid, dec) if not d]
    ftext = {}
    pe = pi = 0
    for i in fid:
        y, a, b = ptd.scrub_pii(cleaned[i]); pe += a; pi += b; ftext[i] = y
    C["pii"] = {"emailsReplaced": pe, "ipsReplaced": pi}
    C["finalDocs"] = len(fid)
    C["finalBytes"] = int(sum(len(ftext[i].encode()) for i in fid))
    hold_set = set(range(n_hold))
    f_hold = [i for i in fid if i in hold_set]
    f_train = [i for i in fid if i not in hold_set]
    # Quality classifier scores on F (train side); E = top half
    sc = score([ftext[i] for i in f_train]) if f_train else np.zeros(0)
    cut = float(np.median(sc)) if len(sc) else 0.0
    e_train = [i for i, s in zip(f_train, sc) if s >= cut]
    C["classifierOnF"] = {"median": round(cut, 4), "fracAtLeast3": round(float(np.mean(sc >= 3)), 4) if len(sc) else None,
                          "fracAtLeast2": round(float(np.mean(sc >= 2)), 4) if len(sc) else None, "eDocs": len(e_train)}
    # Other domains
    wtr = pd.read_parquet(dl(WIKI_TR, "wiki_train0.parquet"))["text"].tolist()
    wte = pd.read_parquet(dl(WIKI_TE, "wiki_test.parquet"))["text"].tolist()

    def articles(lines):
        arts, cur = [], []
        for ln in lines:
            if re.match(r"^ = [^=].* = $", ln.rstrip("\n")) and cur:
                arts.append("".join(cur).strip()); cur = []
            cur.append(ln)
        if cur:
            arts.append("".join(cur).strip())
        return [a for a in arts if len(a) > 200]
    wiki_tr, wiki_te = articles(wtr), articles(wte)
    tstr = pd.read_parquet(dl(TS_TR, "ts_train0.parquet"))["text"].tolist()
    tsva = pd.read_parquet(dl(TS_VA, "ts_val.parquet"))["text"].tolist()
    if SMALL:
        wiki_tr, tstr = wiki_tr[:300], tstr[:3000]
    else:
        tstr = tstr[:160000]
    # Tokenizer: 4,096-token byte-level BPE on a fixed sample of the three domains
    samp = utrain[: (2000 if SMALL else 12000)] + wiki_tr[: (100 if SMALL else 2500)] + tstr[: (2000 if SMALL else 60000)]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<|eot|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tt = time.time(); tok.train_from_iterator(samp, trainer=trainer)
    STATS["tokenizer"] = {"vocab": tok.get_vocab_size(), "sampleDocs": len(samp), "sampleBytes": int(sum(len(x.encode()) for x in samp)),
                          "trainSec": round(time.time() - tt, 1)}
    eot = tok.token_to_id("<|eot|>")
    tok.save(os.path.join(OUT, "tokenizer.json"))
    tb = np.array([len(tok.decode([i]).encode("utf-8")) if i != eot else 1 for i in range(tok.get_vocab_size())], dtype=np.int64)
    np.save(os.path.join(OUT, "token_bytes.npy"), tb)
    cap = 2_000_000 if SMALL else 60_000_000
    save_corpus("U_train", doc_tokens(tok, utrain[: (3600 if SMALL else 40000)], eot), cap)
    save_corpus("U_val_clean", doc_tokens(tok, [hold[i] for i in range(n_hold) if i not in set(leak_idx)], eot), 600_000)
    save_corpus("U_val_leak", doc_tokens(tok, [hold[i] for i in leak_idx], eot), 600_000)
    save_corpus("U_val", doc_tokens(tok, hold, eot), 600_000)
    ftr_docs = doc_tokens(tok, [ftext[i] for i in f_train], eot)
    save_corpus("F_train", ftr_docs)
    fh_docs = doc_tokens(tok, [ftext[i] for i in f_hold], eot)
    # Canaries: 160 held-out filtered documents with at least 100 tokens, first 100 tokens each
    can_idx = [k for k, d in enumerate(fh_docs) if d.size >= 101][:160]
    can = [fh_docs[k][:100] for k in can_idx]
    np.save(os.path.join(OUT, "canaries.npy"), np.stack(can).astype(np.uint16) if can else np.zeros((0, 100), np.uint16))
    rest = [d for k, d in enumerate(fh_docs) if k not in set(can_idx)]
    STATS["canaries"] = {"count": len(can), "tokens": 100}
    save_corpus("F_val", rest, 600_000)
    save_corpus("E_train", doc_tokens(tok, [ftext[i] for i in e_train], eot))
    save_corpus("wiki_train", doc_tokens(tok, wiki_tr, eot), cap)
    save_corpus("wiki_test", doc_tokens(tok, wiki_te, eot), 600_000)
    save_corpus("ts_train", doc_tokens(tok, tstr, eot), cap)
    save_corpus("ts_val", doc_tokens(tok, tsva, eot), 600_000)
    # HellaSwag validation, tokenized for the GPU kernel
    hs = HS
    items = []
    for c, ends, lab in zip(hs["ctx"], hs["endings"], hs["label"]):
        items.append({"ctx": tok.encode(c).ids, "endings": [tok.encode(" " + e).ids for e in ends], "label": int(lab)})
    with open(os.path.join(OUT, "hellaswag_val_tok.json"), "w") as f:
        json.dump(items, f)
    STATS["timings"]["partCSec"] = round(time.time() - t, 1)
    STATS["meta"] = {"eot": eot}
    save()


if __name__ == "__main__":
    dl(LID, "lid.176.bin")
    DECON, HS = eval_sets()
    with Pool(int(os.environ.get("PTD_PROCS", "4"))) as pool:
        score = part_b()
        if "--skip-a" not in sys.argv:
            part_a(pool, DECON, score)
        if "--only-a" not in sys.argv:
            part_c(pool, DECON, score)
    save()
    log("DONE")
