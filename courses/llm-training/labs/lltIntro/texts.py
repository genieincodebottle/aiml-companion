"""The three texts lltIntro measures. The English and Hindi paragraphs say the same thing,
so tokens per word compares the two languages on the same content."""

ENGLISH = (
    "The old library stood at the end of a quiet street. Every morning, the librarian opened "
    "the doors at nine and watered the plants by the window. Children came after school to read "
    "stories, and older readers sat near the back with newspapers. On rainy days the reading room "
    "was full, and the smell of old paper filled the air."
)

HINDI = (
    "पुराना पुस्तकालय एक शांत गली के आखिर में था। हर सुबह लाइब्रेरियन नौ बजे दरवाज़े खोलती थी "
    "और खिड़की के पास रखे पौधों को पानी देती थी। बच्चे स्कूल के बाद कहानियाँ पढ़ने आते थे, और बड़े "
    "पाठक अख़बार लेकर पीछे बैठते थे। बारिश के दिनों में पढ़ने का कमरा भरा रहता था, और हवा में "
    "पुराने कागज़ की खुशबू फैली रहती थी।"
)

PYTHON = '''def moving_average(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    result = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        if i >= window - 1:
            result.append(total / window)
    return result
'''

TEXTS = {"English": ENGLISH, "Python code": PYTHON, "Hindi": HINDI}
PROMPT = "The old library stood at the end of"
