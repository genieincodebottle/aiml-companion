STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}

def gopher_rule(text):
    # Return None to keep the page, or the name of the first rule it fails, checked in this order.
    # word_count: keep 50 to 100,000 words
    # mean_word_length: keep a mean word length of 3 to 10 characters
    # alpha_words: keep pages where at least 80% of words contain a letter
    # stop_words: keep pages with at least 2 of the words in STOP
    # duplicate_lines: remove pages where more than 30% of lines repeat an earlier line
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
