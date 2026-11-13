# ============================================================
# PRODUCTION RAG PIPELINE
# LangChain 1.x | ChromaDB | Google Gemini | FlashRank Reranker
# ============================================================
# pip install -r requirements.txt
# ============================================================

import hashlib
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
# No langchain-community and no langchain-classic: langchain-community is being sunset
# (github.com/langchain-ai/langchain-community/issues/674) and langchain-classic holds
# legacy components. Loading is plain pathlib below, Chroma comes from its partner package,
# and two-stage retrieval lives in src/reranking.py.
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.reranking import FlashRankReranker, RerankingRetriever
from src.security.sanitizer import detect_pii, filter_output_pii, sanitize_input

load_dotenv()


# ---- Step 1: Load Documents ----
def load_documents(data_dir: str = "data/sample_docs/") -> list:
    """Load .txt and .md files from a directory.

    The glob used to be `**/*.txt` while this docstring promised .md as well,
    so markdown dropped into the corpus was skipped in silence -- no error, no
    warning, just answers that could not cite documents the user believed were
    indexed. An empty corpus is loud; a quietly partial one is not.
    """
    raw_docs = []
    for pattern in ("**/*.txt", "**/*.md"):
        # Sorted, so every run indexes the same files in the same order. UTF-8 explicitly:
        # the old TextLoader used the platform default, which is cp1252 on Windows.
        for path in sorted(Path(data_dir).glob(pattern)):
            if path.is_file():
                raw_docs.append(Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": str(path)},
                ))

    if not raw_docs:
        raise FileNotFoundError(
            f"No .txt or .md files found under {data_dir!r}. Indexing an empty "
            f"corpus would build a vector store that answers every question "
            f"with nothing.")
    print(f"Loaded {len(raw_docs)} documents")
    return raw_docs


# ---- Step 2: Chunk Documents ----
def chunk_documents(docs: list, chunk_size: int = 512, chunk_overlap: int = 50) -> list:
    """Split documents into chunks with semantic boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    avg_len = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
    print(f"Created {len(chunks)} chunks (avg {avg_len} chars)")
    return chunks


# ---- Step 3: Create Embeddings + Vector Store ----
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"

# Used only if configs/base.yaml is missing. Keep equal to the shipped config.
DEFAULT_EMBEDDING_CONFIG = {"model": "models/gemini-embedding-001", "dimensions": 3072}


def _load_config_block(name: str, defaults: dict, path: Path = CONFIG_PATH) -> dict:
    """One block of configs/base.yaml laid over its defaults (defaults if the file is missing)."""
    cfg = dict(defaults)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg.update((yaml.safe_load(f) or {}).get(name) or {})
    return cfg


def load_embedding_config(path: Path = CONFIG_PATH) -> dict:
    """Read the `embedding:` block of configs/base.yaml (model + dimensions).

    `dimensions` used to be documented as 768 while nothing read it, so the
    index was silently built at the model's 3072-dim default. It is now read
    here and passed to the embedder as `output_dimensionality`, so the config
    value IS the index dimension. Changing it changes the vector size: delete
    the persisted `chroma_db/` and re-index, or Chroma rejects the mismatch.
    """
    return _load_config_block("embedding", DEFAULT_EMBEDDING_CONFIG, path)


# The reranker model is now explicit. The code used to pass only top_n, so FlashRank ran
# its library default (ms-marco-MultiBERT-L-12) while the config and docs named
# ms-marco-MiniLM-L-12-v2. Every published score was measured with MultiBERT, so that is
# the pinned model; a library upgrade can no longer swap it underneath the results.
DEFAULT_RERANKER_CONFIG = {"model": "ms-marco-MultiBERT-L-12", "provider": "flashrank"}


def load_reranker_config(path: Path = CONFIG_PATH) -> dict:
    """Read the `reranker:` block of configs/base.yaml (FlashRank model name)."""
    return _load_config_block("reranker", DEFAULT_RERANKER_CONFIG, path)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """The embedder used for both indexing and querying (configs/base.yaml)."""
    cfg = load_embedding_config()
    return GoogleGenerativeAIEmbeddings(
        model=cfg["model"],
        output_dimensionality=int(cfg["dimensions"]),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def chunk_id(chunk) -> str:
    """A stable ID derived from the chunk's own content and source.

    The same chunk always hashes to the same ID, which is what turns indexing
    from "append" into "upsert" -- see build_vectorstore for why that matters.
    """
    source = chunk.metadata.get("source", "")
    return hashlib.sha256(f"{source}::{chunk.page_content}".encode("utf-8")).hexdigest()


def build_vectorstore(chunks: list, persist_dir: str = "./chroma_db") -> Chroma:
    """Create (or update) the ChromaDB vector store from document chunks.

    INDEXING MUST BE IDEMPOTENT, and this is the bug that taught it.

    `Chroma.from_documents(..., persist_directory=...)` APPENDS to whatever
    collection is already on disk. It does not replace it. So every re-run of
    the pipeline added a second, third, fourth copy of every chunk, and nothing
    anywhere reported a problem: no error, no warning, and a vector store that
    looks healthy because it is merely larger.

    What it does to retrieval is not subtle. The committed store here had
    accumulated **54 rows for 9 distinct chunks** -- six identical copies of
    everything. Because duplicates have identical embeddings they score
    identically, so a similarity search for the top 20 returns the same handful
    of chunks over and over, the reranker faithfully ranks those duplicates,
    and the top 5 handed to the model were **5 copies of one chunk**:

        query "What is the refund policy?"  -> returned 5, distinct 1
        query "What are the rate limits?"   -> returned 5, distinct 1

    A retrieval system that returns one document is not a retrieval system. The
    context window fills with the same paragraph repeated, the other eight
    chunks become unreachable, and you pay for the tokens.

    The fix is to give each chunk a deterministic ID derived from its content,
    so re-indexing UPSERTS rather than appends. Re-running is now free and the
    collection size stays equal to the number of distinct chunks.
    """
    embeddings = get_embeddings()
    ids = [chunk_id(c) for c in chunks]
    n_unique = len(set(ids))
    if n_unique != len(ids):
        # Identical text in two places is legitimate (a shared boilerplate
        # paragraph); say so rather than silently collapsing them.
        print(f"note: {len(ids) - n_unique} chunks are byte-identical to another "
              f"chunk and will share an ID")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="product_docs",
        ids=ids,
    )
    total = vectorstore._collection.count()
    print(f"Indexed {n_unique} unique chunks; collection now holds {total}")
    if total > n_unique:
        print(f"WARNING: the collection holds {total} rows for {n_unique} unique "
              f"chunks, so it still carries duplicates from an older run. "
              f"Delete '{persist_dir}' and re-index to clear them.")
    return vectorstore


# ---- Step 4: Build Retriever with Reranking ----
def build_retriever(vectorstore: Chroma, use_reranking: bool = True, top_k: int = 20, top_n: int = 5):
    """Build retriever with FlashRank reranking (runs locally, no API key)."""
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    if use_reranking:
        reranker = FlashRankReranker(model=load_reranker_config()["model"], top_n=top_n)
        return RerankingRetriever(base_retriever=base_retriever, reranker=reranker)
    else:
        base_retriever.search_kwargs["k"] = top_n
        return base_retriever


# ---- Step 5: RAG Chain with Citations ----
SYSTEM_PROMPT = """You are an expert assistant. Answer questions ONLY
using the provided context.

Rules:
1. If the context contains the answer, provide it with [Source N] citations
2. If the context partially answers, state what you can confirm and what's missing
3. If the context doesn't contain the answer, say: "I don't have enough
   information in the provided documents to answer this question."
4. NEVER use your training knowledge to fill gaps
5. Rate confidence: HIGH / MEDIUM / LOW

Context:
{context}
"""


def format_docs_with_sources(docs: list) -> str:
    """Format retrieved documents with source citations."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source {i+1}] ({source}):\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(retriever, model: str = "gemini-3.5-flash-lite"):
    """Build the RAG chain with citations.

    No `temperature`: for Gemini 3.5 models Google marks temperature, top_p and top_k as
    deprecated and IGNORED, and future models reject them with HTTP 400. Setting it to 0
    never made answers deterministic here. Consistency comes from the rule-based system
    prompt, which is what Google recommends instead.
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    rag_chain = (
        {"context": retriever | format_docs_with_sources,
         "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain


def query_pipeline(rag_chain, retriever, question: str) -> dict:
    """Query the RAG pipeline through all three security layers.

    1. Input sanitisation: known injection phrases are replaced with [BLOCKED] BEFORE the
       question reaches retrieval or the model. It reads only the question, so it cannot
       catch an instruction that arrives inside a retrieved document (indirect injection);
       the grounded system prompt is the layer for that.
    2. PII detection on the question and on every retrieved chunk. Detection reports what
       it found (types only, never the values) so a caller can log or refuse.
    3. Output filtering: PII in the answer is replaced with typed markers such as
       [EMAIL_REDACTED] before the answer is returned.

    These functions used to exist and be tested but were never called on the query path,
    while the docs said they ran on every query. They now do.
    """
    safe_question = sanitize_input(question)
    response = rag_chain.invoke(safe_question)
    docs = retriever.invoke(safe_question)

    raw_answer = response.text
    answer = filter_output_pii(raw_answer)

    return {
        "question": question,
        "sanitized_question": safe_question,
        "input_blocked": safe_question != question,
        "pii_detected": {
            "question": sorted({kind for kind, _ in detect_pii(question)}),
            "context": sorted({kind for doc in docs for kind, _ in detect_pii(doc.page_content)}),
        },
        "answer_redacted": answer != raw_answer,
        "answer": answer,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "score": doc.metadata.get("relevance_score", None),
                "snippet": doc.page_content[:200],
            }
            for doc in docs
        ],
        "num_sources": len(docs),
    }


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    retriever = build_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)

    question = "What is the refund policy for enterprise customers?"
    result = query_pipeline(rag_chain, retriever, question)

    print("=" * 60)
    print(f"Q: {result['question']}")
    print(f"\nA: {result['answer']}")
    print(f"\nRetrieved {result['num_sources']} sources:")
    for i, src in enumerate(result["sources"]):
        print(f"  [{i+1}] {src['source']} (score: {src['score']})")
        print(f"      {src['snippet']}...")
    print("=" * 60)
