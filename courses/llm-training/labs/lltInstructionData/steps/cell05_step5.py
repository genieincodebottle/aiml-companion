import re
from transformers import pipeline

# Patterns written without backslashes: [0-9] for a digit, [.] for a literal dot.
PATTERNS = {
    "email": re.compile("[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+([.][A-Za-z0-9-]+)*[.][A-Za-z]{2,}"),
    "phone": re.compile("(?<![0-9])[+]?[0-9]{1,3}?[ .(-]{0,2}[0-9]{3}[ .)-]{0,2}[0-9]{3}[ .-]?[0-9]{4}(?![0-9])"),
    "ipv4": re.compile("(?<![0-9.])(?:[0-9]{1,3}[.]){3}[0-9]{1,3}(?![0-9.])"),
    "card": re.compile("(?<![0-9])(?:[0-9][ -]?){13,16}(?![0-9])"),
}

def luhn_ok(digits):
    total = 0
    for k, d in enumerate(reversed(digits)):
        d = int(d) * (2 if k % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0

def regex_hits(text):
    found = {}
    for kind, pat in PATTERNS.items():
        ms = [m.group(0) for m in pat.finditer(text)]
        if kind == "card":
            ms = [m for m in ms if luhn_ok([c for c in m if c.isdigit()])]
        if ms:
            found[kind] = len(ms)
    return found

user_rows = [i for i, r in enumerate(pool) if r["source"] == "wildchat"]
pii = {i: regex_hits(pool[i]["prompt"]) for i in user_rows}
pii = {i: h for i, h in pii.items() if h}
by_kind = collections.Counter(k for h in pii.values() for k in h)
print(len(user_rows), "user prompts from WildChat |", len(pii), "with a regex hit |", dict(by_kind))
print("WildChat's own redacted flag on these rows:", sum(bool(pool[i]["redacted"]) for i in user_rows))

# Names: a named-entity recognition (NER) model tags person names in English prompts.
ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple",
               device=0 if torch.cuda.is_available() else -1)
english = [i for i in user_rows if pool[i]["lang"] == "English"]
names = {}
for i, ents in zip(english, ner([pool[i]["prompt"][:1500] for i in english], batch_size=32)):
    people = [e["word"] for e in ents if e["entity_group"] == "PER" and e["score"] >= 0.9]
    if people:
        names[i] = people
print(len(english), "English user prompts |", len(names), "name at least one person (NER score 0.9 or more)")
del ner
