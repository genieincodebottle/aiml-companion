# ============================================================
# PRODUCTION RAG PIPELINE
# LangChain 1.x | ChromaDB | Google Gemini | FlashRank Reranker
# ============================================================
# pip install langchain langchain-google-genai langchain-community
# pip install chromadb flashrank
# ============================================================

import hashlib
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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
        loader = DirectoryLoader(
            data_dir,
            glob=pattern,
            loader_cls=TextLoader,
            show_progress=True,
        )
        raw_docs.extend(loader.load())

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
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
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
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        reranker = FlashrankRerank(top_n=top_n)
        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )
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
    """Build the RAG chain with citations."""
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
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
    """Query the RAG pipeline and return response with sources."""
    response = rag_chain.invoke(question)
    docs = retriever.invoke(question)

    return {
        "question": question,
        "answer": response.text,
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
