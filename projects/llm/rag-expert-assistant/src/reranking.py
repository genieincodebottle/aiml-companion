# ============================================================
# TWO-STAGE RETRIEVAL: dense search, then FlashRank reranking
# ============================================================
#
# Why this is local code. The project used two wrappers for this:
#   - langchain_community.document_compressors.FlashrankRerank
#   - langchain_classic.retrievers.ContextualCompressionRetriever
# langchain-community is being sunset and is no longer maintained
# (github.com/langchain-ai/langchain-community/issues/674), and langchain-classic is the
# home of legacy LangChain components. Neither reranker has a standalone partner
# package, so the two small pieces live here, built on langchain-core and flashrank.
#
# The reranking logic is a faithful port of FlashrankRerank.compress_documents: score
# every candidate against the query, keep the top_n at or above score_threshold (0.0 by
# default), and put the score in metadata["relevance_score"].
# ============================================================

from typing import List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


class FlashRankReranker:
    """Cross-encoder reranking with FlashRank. Runs locally on CPU, no API key.

    The model is loaded on first use (FlashRank downloads it once), so building a
    pipeline or importing this module never touches the network.
    """

    def __init__(self, model: str, top_n: int = 5, score_threshold: float = 0.0, ranker=None):
        self.model = model
        self.top_n = top_n
        self.score_threshold = score_threshold
        self._ranker = ranker

    @property
    def ranker(self):
        if self._ranker is None:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=self.model)
        return self._ranker

    def rerank(self, query: str, documents: Sequence[Document]) -> List[Document]:
        if not documents:
            return []
        from flashrank import RerankRequest

        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]
        ranked = self.ranker.rerank(RerankRequest(query=query, passages=passages))[: self.top_n]
        return [
            Document(
                page_content=r["text"],
                metadata={"id": r["id"], "relevance_score": float(r["score"]), **r["meta"]},
            )
            for r in ranked
            if r["score"] >= self.score_threshold
        ]


class RerankingRetriever(BaseRetriever):
    """Retrieve broadly with base_retriever, then keep the reranker's best top_n."""

    base_retriever: BaseRetriever
    reranker: FlashRankReranker
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        config = {"callbacks": run_manager.get_child()} if run_manager else None
        candidates = self.base_retriever.invoke(query, config=config)
        return self.reranker.rerank(query, candidates)
