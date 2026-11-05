# prep_fineweb.py - FineWeb-Edu sample-10BT -> GPT-2 tokens, one uint16 shard per parquet file.
#
#   python prep_fineweb.py --out /workspace/data                     # all 14 files, about 10B tokens
#   python prep_fineweb.py --out /workspace/data --files 1           # calibration: one file
#   python prep_fineweb.py --out /workspace/hq --min-int-score 4     # the ablation's high-score subset
#   python prep_fineweb.py --out data --file-ids 0,12,13 --max-docs 150000   # learner decay path: val + 2 unseen files
#
# Layout: <out>/val.bin (the first --val-tokens tokens of file 000), <out>/train_XXX.bin, <out>/index.json.
# Every document starts with the end-of-text token 50256, as in llm.c. Tokens are stored as uint16 (vocab 50,257).
import argparse, json, os, time
from multiprocessing import Pool

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--files", type=int, default=14)            # sample/10BT has 14 parquet files (checked 2026-11-05)
p.add_argument("--file-ids", default="")                   # e.g. "0,12,13" for the learner path; overrides --files
p.add_argument("--val-tokens", type=int, default=10_485_760)
p.add_argument("--max-docs", type=int, default=0)          # per file; 0 = all (tests use a small number)
p.add_argument("--min-int-score", type=int, default=0)     # FineWeb-Edu classifier score filter; 0 = keep all
p.add_argument("--workers", type=int, default=0)
a = p.parse_args()


def one_file(i):
    import numpy as np
    import pyarrow.parquet as pq
    import tiktoken
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    path = hf_hub_download("HuggingFaceFW/fineweb-edu", f"sample/10BT/{i:03d}_00000.parquet", repo_type="dataset")
    t_dl = time.time() - t0
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    out = os.path.join(a.out, f"raw_{i:03d}.bin")
    n_tok = n_doc = n_chr = 0
    with open(out, "wb") as f:
        cols = ["text", "int_score"] if a.min_int_score else ["text"]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=cols):
            texts = batch.column(0).to_pylist()
            if a.min_int_score:
                texts = [t for t, s in zip(texts, batch.column(1).to_pylist()) if s >= a.min_int_score]
            if a.max_docs:
                texts = texts[: max(0, a.max_docs - n_doc)]
            if not texts:
                if a.max_docs and n_doc >= a.max_docs:
                    break
                continue
            ids = enc.encode_ordinary_batch(texts, num_threads=2)
            arr = np.concatenate([np.array([eot] + x, dtype=np.uint16) for x in ids])
            f.write(arr.tobytes())
            n_tok += len(arr)
            n_doc += len(texts)
            n_chr += sum(len(t) for t in texts)
            if a.max_docs and n_doc >= a.max_docs:
                break
    return {"file": i, "tokens": n_tok, "docs": n_doc, "chars": n_chr, "downloadSec": round(t_dl, 1),
            "sec": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    ids = [int(i) for i in a.file_ids.split(",")] if a.file_ids else list(range(a.files))
    if 0 not in ids:
        raise SystemExit("file 0 is needed: the fixed validation slice comes from it")
    ids = [0] + [i for i in ids if i != 0]
    workers = a.workers or max(1, min(len(ids), (os.cpu_count() or 2) // 2))
    with Pool(workers) as pool:
        parts = pool.map(one_file, ids)
    import numpy as np
    # Split file 000 into the fixed validation slice and its training remainder, then name the train shards.
    raw0 = np.memmap(os.path.join(a.out, "raw_000.bin"), dtype=np.uint16, mode="r")
    if len(raw0) <= a.val_tokens:
        raise SystemExit(f"file 000 has {len(raw0)} tokens, fewer than --val-tokens {a.val_tokens}")
    raw0[: a.val_tokens].tofile(os.path.join(a.out, "val.bin"))
    shards = []
    for part in parts:
        i = part["file"]
        src = os.path.join(a.out, f"raw_{i:03d}.bin")
        dst = os.path.join(a.out, f"train_{i:03d}.bin")
        start = a.val_tokens if i == 0 else 0
        os.replace(src, dst)
        shards.append({"file": os.path.basename(dst), "tokens": part["tokens"], "start": start})
    del raw0
    total = sum(s["tokens"] - s["start"] for s in shards)
    import tiktoken
    index = {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "licence": "ODC-By 1.0",
             "tokenizer": "gpt2 (tiktoken " + tiktoken.__version__ + ")", "eot": 50256, "dtype": "uint16",
             "minIntScore": a.min_int_score, "maxDocsPerFile": a.max_docs,
             "val": {"file": "val.bin", "tokens": a.val_tokens}, "train": shards, "trainTokens": total,
             "docs": sum(q["docs"] for q in parts), "chars": sum(q["chars"] for q in parts),
             "perFile": parts, "workers": workers, "sec": round(time.time() - t0, 1)}
    index["charsPerToken"] = round(index["chars"] / max(1, total + a.val_tokens), 3)
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(json.dumps({k: v for k, v in index.items() if k not in ("perFile", "train")}))
