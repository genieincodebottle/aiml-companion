STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}

def gopher_quality(text):
    words = text.split()
    content = [w for w in words if any(c.isalnum() for c in w)]
    if not 50 <= len(content) <= 100_000:
        return "word count outside 50 to 100,000"
    mean = sum(len(w) for w in content) / len(content)
    if not 3 <= mean <= 10:
        return "mean word length outside 3 to 10"
    if (text.count("#") + text.count("...")) / len(words) > 0.1:
        return "too many # or ..."
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < 0.8:
        return "under 80% of words have a letter"
    if len(STOP & {w.lower() for w in words}) < 2:
        return "fewer than 2 stop words"
    return None

def top_ngram_share(text, n):
    words = text.split()
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    best = max(set(grams), key=lambda g: (grams.count(g), len(g)))
    return len(best) * grams.count(best) / len(text)

story = ("The library opens at nine every morning and closes at six. Children come after school "
         "to read, and older readers sit near the back with the newspapers. On rainy days the "
         "reading room is full, and the staff put out extra chairs for everyone. The building "
         "was restored last year with a new roof and better lighting over the long tables.")
pages = {
    "article": story,
    "menu": "Home About Shop Contact Login Cart Search Sale New Brands",
    "price list": " ".join(f"item{i} 4.99 USD 12 pcs" for i in range(30)),
    "keyword spam": " ".join(["buy the cheap phone cases and save today"] * 9),
    "tag cloud": " ".join(["#deal #sale #shop the best of"] * 12),
}
for name, text in pages.items():
    rule = gopher_quality(text)
    share = top_ngram_share(text, 2)
    if rule is None and share > 0.20:
        rule = "top 2-gram covers %.0f%% of characters" % (100 * share)
    print(f"{name:13} {'kept' if rule is None else 'removed, ' + rule}")
