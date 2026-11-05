import os
import time
import urllib.request
import trafilatura
from warcio.archiveiterator import ArchiveIterator

URL = ("https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/1707947473347.0/"
       "warc/CC-MAIN-20240220211055-20240221001055-00000.warc.gz")
for attempt in range(7):  # Common Crawl answers 503 when busy, so wait and retry
    if os.path.exists("warc_slice.gz") and os.path.getsize("warc_slice.gz") == 62914560:
        break
    try:
        req = urllib.request.Request(URL, headers={"Range": "bytes=0-62914559", "User-Agent": "llt-lab"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        with open("warc_slice.gz", "wb") as f:
            f.write(data)
    except Exception as e:
        print("download failed,", e, "| retrying in", 15 * 2 ** attempt, "s")
        time.sleep(15 * 2 ** attempt)

records, pages = 0, []
try:
    with open("warc_slice.gz", "rb") as f:
        for rec in ArchiveIterator(f):
            records += 1
            h = rec.http_headers
            is_html = h is not None and "html" in (h.get_header("Content-Type") or "").lower()
            if rec.rec_type == "response" and is_html and h.get_statuscode() == "200":
                url = rec.rec_headers.get_header("WARC-Target-URI")
                pages.append((url, rec.content_stream().read().decode("utf-8", "replace")))
except Exception as e:  # the first 60 MB ends in the middle of a record
    print("slice ends mid-record,", type(e).__name__)

docs = []
for url, html in pages:
    text = trafilatura.extract(html, favor_precision=True, include_comments=False, deduplicate=True)
    if text and text.strip():
        docs.append((url, text))
print("WARC records", records, "| HTML pages with status 200", len(pages), "| pages with text", len(docs))
print("HTML characters", sum(len(h) for _, h in pages), "| text characters", sum(len(t) for _, t in docs))
