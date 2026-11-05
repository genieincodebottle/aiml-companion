"""Build the text used by the lltTokenizer lab. Every source is licence-clear and recorded.

  FLORES-200 dev and devtest (English, Hindi, Spanish), CC BY-SA 4.0
      https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
      The same sentences in every language, so tokens per word compares like with like.
  Wikipedia 20231101 (English, Hindi, Spanish), CC BY-SA 3.0 and GFDL
      Hugging Face dataset wikimedia/wikipedia, streamed, first articles in file order.
  Python source code: the CPython standard library of the running interpreter, PSF licence.

Usage: python data.py   (writes to ~/.cache/lltlab/tokenizer, about 30 MB)
"""
import hashlib
import json
import os
import sys
import sysconfig
import tarfile
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("LLT_TOK_CACHE", Path.home() / ".cache" / "lltlab" / "tokenizer"))
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
LANGS = {"en": "eng_Latn", "hi": "hin_Deva", "es": "spa_Latn"}
WIKI_BYTES = 8_000_000      # per language, training text
CODE_TRAIN_BYTES = 4_000_000
CODE_HELDOUT_BYTES = 150_000


def flores():
    out = CACHE / "flores"
    if all((out / f"{c}.{s}").exists() for c in LANGS for s in ("dev", "devtest")):
        return out
    out.mkdir(parents=True, exist_ok=True)
    tgz = CACHE / "flores200_dataset.tar.gz"
    if not tgz.exists():
        urllib.request.urlretrieve(FLORES_URL, tgz)
    digest = hashlib.sha256(tgz.read_bytes()).hexdigest()
    assert digest == FLORES_SHA256, f"FLORES archive changed: {digest}"
    with tarfile.open(tgz) as tf:
        for code, name in LANGS.items():
            for split in ("dev", "devtest"):
                member = tf.getmember(f"./flores200_dataset/{split}/{name}.{split}")
                (out / f"{code}.{split}").write_bytes(tf.extractfile(member).read())
    return out


def wikipedia():
    from datasets import load_dataset
    meta = {}
    for code in LANGS:
        path = CACHE / f"wiki_{code}.txt"
        mpath = CACHE / f"wiki_{code}.json"
        if path.exists() and mpath.exists():
            meta[code] = json.loads(mpath.read_text())
            continue
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{code}", split="train", streaming=True)
        parts, size, n, first = [], 0, 0, None
        for ex in ds:
            first = first or ex["url"]
            parts.append(ex["text"])
            size += len(ex["text"].encode("utf-8"))
            n += 1
            if size >= WIKI_BYTES:
                break
        path.write_bytes("\n\n".join(parts).encode("utf-8"))
        meta[code] = {"articles": n, "bytes": size, "firstUrl": first, "lastUrl": ex["url"]}
        mpath.write_text(json.dumps(meta[code]))
    return meta


def python_code():
    lib = Path(sysconfig.get_paths()["stdlib"])
    skip = ("site-packages", "test", "tests", "idlelib", "lib2to3", "turtledemo")
    files = sorted(p for p in lib.rglob("*.py") if not any(s in p.relative_to(lib).parts for s in skip))
    train, held = [], []
    tsize = hsize = 0
    for i, p in enumerate(files):
        text = p.read_text(encoding="utf-8", errors="strict") if p.stat().st_size < 400_000 else ""
        if not text:
            continue
        rel = str(p.relative_to(lib)).replace(os.sep, "/")
        if i % 10 == 0 and hsize < CODE_HELDOUT_BYTES:
            held.append((rel, text)); hsize += len(text.encode("utf-8"))
        elif i % 10 != 0 and tsize < CODE_TRAIN_BYTES:
            train.append((rel, text)); tsize += len(text.encode("utf-8"))
    # write_bytes, not write_text: on Windows write_text turns every newline into CRLF.
    (CACHE / "code_train.txt").write_bytes("\n\n".join(t for _, t in train).encode("utf-8"))
    (CACHE / "code_heldout.txt").write_bytes("\n\n".join(t for _, t in held).encode("utf-8"))
    meta = {"python": sys.version.split()[0], "trainFiles": len(train), "trainBytes": tsize,
            "heldoutFiles": [r for r, _ in held], "heldoutBytes": hsize}
    (CACHE / "code.json").write_text(json.dumps(meta, indent=1))
    return meta


def build():
    CACHE.mkdir(parents=True, exist_ok=True)
    flores()
    wiki = wikipedia()
    code = python_code() if not (CACHE / "code.json").exists() else json.loads((CACHE / "code.json").read_text())
    return {"wikipedia": wiki, "code": code}


def _read(name):
    path = CACHE / name
    if not path.exists():
        build()
    return path.read_bytes().decode("utf-8")


# Helpers the notebook cells and measure.py share, so both read the same text the same way.
def heldout():
    """FLORES-200 devtest per language ('en', 'hi', 'es') and held-out Python code ('code')."""
    out = {code: _read(f"flores/{code}.devtest") for code in LANGS}
    out["code"] = _read("code_heldout.txt")
    return out


def train_texts():
    """Training text per source: Wikipedia 'en', 'hi', 'es' and Python 'code'."""
    out = {code: _read(f"wiki_{code}.txt") for code in LANGS}
    out["code"] = _read("code_train.txt")
    return out


def train_slice(n_bytes=250_000):
    """The first n_bytes of each training source, joined: 1 MB for the default."""
    parts = [t.encode("utf-8")[:n_bytes].decode("utf-8", "ignore") for t in train_texts().values()]
    return "\n".join(parts)


def lines(texts):
    """Every line of every text, the unit the trainers and the counts use."""
    for text in texts:
        yield from text.splitlines()


def heldout_texts():
    """The 3,682 texts of the parity check: FLORES lines, then code chunks (files split on blank lines)."""
    ho = heldout()
    texts = [line for code in LANGS for line in ho[code].splitlines() if line]
    return texts + [c for c in ho["code"].split("\n\n") if c]


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    print("FLORES ->", flores())
    print("Wikipedia ->", wikipedia())
    m = python_code()
    print("Code ->", m["trainFiles"], "train files,", len(m["heldoutFiles"]), "held-out files")
