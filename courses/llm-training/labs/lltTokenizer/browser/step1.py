samples = {
    "English": "Tokenizers turn text into numbers.",
    "Hindi": "टोकनाइज़र टेक्स्ट को संख्याओं में बदलता है।",
    "Spanish": "Los tokenizadores convierten el texto en números.",
    "Python": "def add(a, b): return a + b",
}
for name, text in samples.items():
    data = text.encode("utf-8")
    print(f"{name:8} {len(text):3} characters {len(data):3} bytes "
          f"{len(data) / len(text):.2f} bytes per character")

print(list("n".encode("utf-8")), list("न".encode("utf-8")), list("ñ".encode("utf-8")))
ids = list(samples["Hindi"].encode("utf-8"))
print(bytes(ids).decode("utf-8") == samples["Hindi"])
