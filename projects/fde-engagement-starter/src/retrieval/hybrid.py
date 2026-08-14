"""Hybrid retrieval over Northwind Freight's policy corpus.

YOU IMPLEMENT THIS. See tests/test_retrieval.py for the contract.

Why hybrid rather than "just use embeddings":

Enterprise corpora are full of identifiers, product codes, policy numbers and
acronyms. Dense retrieval is good at meaning and bad at exact tokens, so a query
for policy "NF-114" can return five documents about the right topic and not the
one with that string in it. Lexical retrieval (BM25) has the opposite failure.
Hybrid is not a sophistication flex, it is the fix for a specific bug you will hit
on day two of almost every engagement.

The contract the tests enforce:

1. Each stage can be switched on and off independently. You cannot claim hybrid
   beats the baseline unless you can run the baseline.
2. `evaluate_recall` produces a comparable number for any configuration, on the
   customer's actual corpus.
3. Hybrid beats dense-only on the corpus in `customer/data/policies/`. If it does
   not for you, that is a finding worth investigating rather than a test to skip.

The test does not care HOW you fuse the two rankings. Reciprocal rank fusion is
the usual answer and score normalisation is a defensible one. What the test cares
about is that you measured it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class RetrievalConfig:
    """Which stages run.

    Keeping this as data rather than three separate functions is what makes the
    ablation cheap, and an ablation you can run in one line is an ablation you
    will actually run.
    """

    use_lexical: bool = True
    use_dense: bool = True
    use_reranker: bool = False
    top_k: int = 5


def load_corpus(policies_dir: Path) -> list[Document]:
    """Load the policy documents into Documents.

    Implementer notes:
        - Chunking is a decision, not a detail. These documents are short enough
          to index whole; say why you chose what you chose.
        - Keep doc_id stable and human readable. You will be citing it in a demo,
          and "NF-SLA-2024.md" reads better than "chunk_17".
    """
    raise NotImplementedError("Implement load_corpus. See tests/test_retrieval.py.")


def search(
    query: str, corpus: list[Document], config: RetrievalConfig
) -> list[tuple[Document, float]]:
    """Return the top_k documents for a query, ranked, with scores.

    Args:
        query: the user question.
        corpus: documents from load_corpus.
        config: which stages to run.

    Returns:
        Up to `config.top_k` (Document, score) pairs, best first.

    Implementer notes:
        - `rank_bm25` is in requirements.txt for the lexical half.
        - For the dense half, `sklearn`'s TfidfVectorizer with cosine similarity is
          a legitimate offline stand-in. It needs no API key and no GPU, so the
          test suite stays runnable on a customer laptop with no internet. Swapping
          in real embeddings later is a contained change, which is itself the
          lesson: keep the retrieval interface stable and the backend swappable.
        - If both stages are off, raise ValueError. A retriever that silently
          returns nothing is a debugging afternoon you do not need.
    """
    raise NotImplementedError("Implement search. See tests/test_retrieval.py.")


def evaluate_recall(
    queries: list[tuple[str, str]], corpus: list[Document], config: RetrievalConfig
) -> float:
    """Recall@k for a configuration.

    Args:
        queries: (query, expected_doc_id) pairs.
        corpus: the corpus.
        config: the configuration to measure.

    Returns:
        Fraction of queries where the expected doc appeared in the top k.

    This is the function that lets you say "hybrid moved recall at 5 from 0.6 to
    0.9 on their corpus" instead of "hybrid is better". One of those sentences
    survives a cross-question.
    """
    raise NotImplementedError("Implement evaluate_recall. See tests/test_retrieval.py.")
