"""The fixed prompt set for lltBaseVsInstruct (40 prompts, written for this lab).

Five groups of 8: how-to and advice, explanations, code, writing, maths and reasoning.
The order is fixed, and every Run Library file records which prompts it used by index.
"""

PROMPTS = [
    # how-to and advice
    "How do I make a cup of green tea?",
    "Give me three tips for sleeping better.",
    "How should I prepare for a job interview?",
    "What is a good way to start learning to play the guitar?",
    "How can I keep my houseplants alive?",
    "What should I pack for a weekend hiking trip?",
    "How do I write a polite email asking for a deadline extension?",
    "How can I save money on groceries?",
    # explanations
    "Why is the sky blue?",
    "Explain how a vaccine works.",
    "What causes the seasons on Earth?",
    "What is the difference between weather and climate?",
    "How does a refrigerator keep food cold?",
    "What is photosynthesis?",
    "Why do we have leap years?",
    "What is inflation in economics?",
    # code
    "Write a Python function that reverses a string.",
    "How do I read a CSV file in Python?",
    "What is the difference between a list and a tuple in Python?",
    "Write a Python function that checks whether a number is prime.",
    "Explain what a for loop does.",
    "How do I remove duplicates from a list in Python?",
    "Write a SQL query that counts the rows in a table called orders.",
    "What does the git commit command do?",
    # writing
    "Write a short poem about the ocean.",
    "Write a two-sentence story about a lost dog.",
    "Write a thank-you note to a teacher.",
    "Suggest a name for a small bakery.",
    "Describe a sunset in three sentences.",
    "Write a short product description for a reusable water bottle.",
    "Write a haiku about autumn.",
    "Write an opening line for a mystery novel.",
    # maths and reasoning
    "What is 15 percent of 80?",
    "If a train travels 60 km in 45 minutes, what is its speed in km per hour?",
    "What is the next number in the sequence 2, 4, 8, 16?",
    "A shirt costs 20 dollars after a 20 percent discount. What was the original price?",
    "How many minutes are there in a week?",
    "Is 91 a prime number?",
    "What is the square root of 144?",
    "If I have 3 apples and eat one, how many are left?",
]

assert len(PROMPTS) == 40
