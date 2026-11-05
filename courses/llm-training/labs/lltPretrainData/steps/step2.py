import collections
import urllib.request
import fasttext

urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", "lid.176.bin")
lid = fasttext.load_model("lid.176.bin")

def language(text):
    labels, probs = lid.predict(text.replace(chr(10), " ")[:20000], k=1)
    return labels[0].replace("__label__", ""), float(probs[0])

langs = [language(text) for _, text in docs]
print(collections.Counter(code for code, _ in langs).most_common(6))
english = [d for d, (code, p) in zip(docs, langs) if code == "en" and p >= 0.65]
unsure = sum(code == "en" and p < 0.65 for code, p in langs)
print("English with score >= 0.65", len(english), "| English below 0.65", unsure)
