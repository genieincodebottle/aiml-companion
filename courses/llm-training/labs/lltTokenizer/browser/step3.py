from collections import Counter
from itertools import groupby

def kind(ch):
    return "L" if ch.isalpha() else "N" if ch.isnumeric() else "S" if ch.isspace() else "P"

def pretokenize(text):
    runs = [[k, "".join(g)] for k, g in groupby(text, kind)]
    pieces, lead = [], ""
    for i, (k, run) in enumerate(runs):
        last = i == len(runs) - 1
        if not run:
            continue
        if k == "S" and not last:
            if len(run) > 1:
                pieces.append(run[:-1])
            if run[-1] == " ":
                lead = " "
            else:
                pieces.append(run[-1])
            continue
        if run == "'" and not lead and not last and runs[i + 1][0] == "L":
            nxt = runs[i + 1][1]
            c = next((c for c in ("s", "t", "re", "ve", "m", "ll", "d") if nxt.startswith(c)), "")
            if c:
                pieces.append("'" + c)
                runs[i + 1][1] = nxt[len(c):]
                continue
        pieces.append(lead + run)
        lead = ""
    return pieces

print(pretokenize("It's 2026, and x  = 10;"))
print(pretokenize("टोकन बनता है।"))
text = "The tokenizer reads text as bytes. El tokenizador lee el texto como bytes."
pieces = pretokenize(text)
print("".join(pieces) == text, len(pieces), "pieces")

raw = Counter(zip(text, text[1:]))
inside = Counter(pair for p in pieces for pair in zip(p, p[1:]))
ends_word = lambda pairs: sum(n for (a, b), n in pairs.items() if b == " " and a != " ")
print("pairs that end a word with a space: raw text", ends_word(raw), "| inside pieces", ends_word(inside))
