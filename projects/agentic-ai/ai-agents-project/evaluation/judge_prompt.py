# ============================================
# Evaluation Prompts for LLM-as-Judge Scoring
# ============================================

ACCURACY_PROMPT = """Score this research report's factual accuracy on a 0-3 scale.

Query: {query}
Report: {report}

Scoring rubric:
- 0 (Missing): No factual claims or entirely fabricated
- 1 (Poor): Multiple factual errors or outdated claims
- 2 (Adequate): Mostly correct with minor inaccuracies
- 3 (Excellent): All claims verifiable and up-to-date

Respond with a single integer (0-3) and one sentence justification.
Format: SCORE: N | REASON: ...
"""

COMPLETENESS_PROMPT = """Score this research report's completeness on a 0-3 scale.

Query: {query}
Report: {report}

Scoring rubric:
- 0 (Missing): Topic not addressed at all
- 1 (Poor): Covers only one aspect, misses key dimensions
- 2 (Adequate): Covers main points but lacks depth or misses nuances
- 3 (Excellent): Comprehensive coverage with multiple perspectives

Respond with a single integer (0-3) and one sentence justification.
Format: SCORE: N | REASON: ...
"""

CITATION_PROMPT = """Score this research report's citation quality on a 0-3 scale.

Query: {query}
Report: {report}

Scoring rubric:
- 0 (Missing): No sources cited at all
- 1 (Poor): Sources mentioned but no URLs or specific references
- 2 (Adequate): Some real sources with URLs, but some may be hallucinated
- 3 (Excellent): All citations link to real, verifiable sources

Respond with a single integer (0-3) and one sentence justification.
Format: SCORE: N | REASON: ...
"""

COMBINED_PROMPT = """Score this research report on a 0-3 scale for each dimension.

Query: {query}
Report: {report}

Score (0=missing, 1=poor, 2=adequate, 3=excellent):
- accuracy: (are claims factually correct?)
- completeness: (are key aspects covered?)
- citations: (are sources provided and real?)

Respond as JSON: {{"accuracy": N, "completeness": N, "citations": N}}
"""