# prep_data.py - download one FineWeb-Edu sample-10BT parquet shard, tokenize with the GPT-2 tokenizer,
# write uint16 token files. Usage: python prep_data.py <out_dir> <train_tokens> <val_tokens>
# v2: v1 streamed with datasets and did not finish inside 900 s on Kaggle, so this reads one shard directly.
import json, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

out_dir, n_train, n_val = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token                                    # 50256, put before every document as in llm.c

t0 = time.time()
path = hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", repo_type="dataset")
dl_sec, shard_bytes = time.time() - t0, os.path.getsize(path)
print("downloaded", round(shard_bytes / 1e9, 2), "GB in", round(dl_sec, 1), "s", flush=True)

t1 = time.time()
need = n_train + n_val
buf, have, docs, chars = [], 0, 0, 0
pf = pq.ParquetFile(path)
for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
    texts = batch.column(0).to_pylist()
    for ids in enc.encode_ordinary_batch(texts, num_threads=4):
        buf.append(np.array([EOT] + ids, dtype=np.uint16))
        have += len(ids) + 1
    docs += len(texts)
    chars += sum(len(t) for t in texts)
    if have >= need:
        break
tok_sec = time.time() - t1
toks = np.concatenate(buf)
val, train = toks[:n_val], toks[n_val:n_val + n_train]   # first documents are the validation slice
train.tofile(f"{out_dir}/train.bin")
val.tofile(f"{out_dir}/val.bin")
info = {"shard": "sample/10BT/000_00000.parquet", "shardGB": round(shard_bytes / 1e9, 3), "shardRows": pf.metadata.num_rows,
        "downloadSec": round(dl_sec, 1), "docs": docs, "chars": chars, "tokens": int(len(toks)),
        "train_tokens": int(len(train)), "val_tokens": int(len(val)), "chars_per_token": round(chars / len(toks), 3),
        "tokenizeSec": round(tok_sec, 1), "tokPerSecTokenize": round(len(toks) / tok_sec), "tiktoken": tiktoken.__version__}
print(json.dumps(info), flush=True)
os._exit(0)
