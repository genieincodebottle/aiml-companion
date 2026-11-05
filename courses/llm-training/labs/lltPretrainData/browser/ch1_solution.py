STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}

def gopher_rule(text):
    # Return None to keep the page, or the name of the first rule it fails.
    words = text.split()
    if not 50 <= len(words) <= 100_000:
        return "word_count"
    mean = sum(len(w) for w in words) / len(words)
    if not 3 <= mean <= 10:
        return "mean_word_length"
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < 0.8:
        return "alpha_words"
    if len(STOP & {w.lower() for w in words}) < 2:
        return "stop_words"
    lines = text.splitlines()
    if len(set(lines)) < 0.7 * len(lines):
        return "duplicate_lines"
    return None

# Checks. Run the cell. It prints All checks passed when gopher_rule is right.
good = " ".join(["The museum opens at ten and the guides lead a tour of the old halls."] * 4)
cases = [
    (good, None),
    ("Buy now. Free shipping.", "word_count"),
    (" ".join(["a an I"] * 30), "mean_word_length"),
    (" ".join(["1990 2001 2024 3000 the of and cat"] * 10), "alpha_words"),
    (" ".join(["cheap shoes sale online red blue black white"] * 10), "stop_words"),
    (chr(10).join(["Subscribe to the newsletter and the offers"] * 12 + [good]), "duplicate_lines"),
    (" ".join(["supercalifragilisticexpialidocious extraordinarily the of"] * 20), "mean_word_length"),
]
for text, want in cases:
    got = gopher_rule(text)
    assert got == want, f"expected {want} for {text[:40]!r}, got {got}"
print("kept", sum(gopher_rule(t) is None for t, _ in cases), "of", len(cases))
print("All checks passed")
