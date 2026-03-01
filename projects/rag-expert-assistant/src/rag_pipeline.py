# ============================================================
# PRODUCTION RAG PIPELINE
# LangChain 0.3+ | ChromaDB | OpenAI | Cohere Reranker
# ============================================================
# pip install langchain langchain-openai langchain-community
# pip install chromadb cohere sentence-transformers
# ============================================================

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()


# ---- Step 1: Load Documents ----
def load_documents(data_dir: str = "data/sample_docs/") -> list:
    """Load all .txt and .md files from a directory."""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    raw_docs = loader.load()
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
def build_vectorstore(chunks: list, persist_dir: str = "./chroma_db") -> Chroma:
    """Create ChromaDB vector store from document chunks."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="product_docs"
    )
    print(f"Indexed {len(chunks)} chunks in ChromaDB")
    return vectorstore


# ---- Step 4: Build Retriever with Reranking ----
def build_retriever(vectorstore: Chroma, use_reranking: bool = True, top_k: int = 20, top_n: int = 5):
    """Build retriever with optional Cohere reranking."""
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    if use_reranking and os.getenv("COHERE_API_KEY"):
        from langchain_community.document_compressors import CohereRerank
        reranker = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            top_n=top_n,
            model="rerank-v3.5"
        )
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


def build_rag_chain(retriever, model: str = "gpt-4o-mini"):
    """Build the RAG chain with citations."""
    llm = ChatOpenAI(model=model, temperature=0)
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
        "answer": response.content,
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