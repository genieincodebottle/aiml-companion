#!/usr/bin/env python3
"""Initialize the research cache with sample data.

Seeds the SQLite cache with example queries so users can explore
cache functionality (hit counting, source indexing) immediately.

Run: python scripts/initialize_cache.py
"""

import sys
import os

# Add project root to path so src imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cache.research_cache import cache_sources, get_cache_stats

SAMPLE_QUERIES = {
    "transformer architecture attention mechanism": [
        {"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need", "snippet": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."},
        {"url": "https://jalammar.github.io/illustrated-transformer/", "title": "The Illustrated Transformer", "snippet": "A visual walkthrough of the Transformer model architecture and self-attention mechanism."},
    ],
    "multi-agent collaboration patterns": [
        {"url": "https://arxiv.org/abs/2308.08155", "title": "AutoGen: Enabling Next-Gen LLM Applications", "snippet": "A framework for building LLM applications using multiple conversable agents."},
        {"url": "https://lilianweng.github.io/posts/2023-06-23-agent/", "title": "LLM Powered Autonomous Agents", "snippet": "An overview of building autonomous agents with LLMs as the core controller."},
        {"url": "https://www.anthropic.com/research/building-effective-agents", "title": "Building Effective Agents", "snippet": "Practical patterns for orchestrating multiple AI agents in production systems."},
    ],
    "retrieval augmented generation RAG best practices": [
        {"url": "https://arxiv.org/abs/2312.10997", "title": "Retrieval-Augmented Generation for Large Language Models: A Survey", "snippet": "A comprehensive survey of RAG methods, covering retrieval strategies and generation techniques."},
        {"url": "https://docs.llamaindex.ai/en/stable/", "title": "LlamaIndex Documentation", "snippet": "A data framework for connecting custom data sources to large language models."},
    ],
    "LLM evaluation metrics and benchmarks": [
        {"url": "https://arxiv.org/abs/2307.03109", "title": "Judging LLM-as-a-Judge", "snippet": "Explores the use of strong LLMs as judges to evaluate other model outputs."},
        {"url": "https://huggingface.co/spaces/open-llm-leaderboard", "title": "Open LLM Leaderboard", "snippet": "Community-driven leaderboard tracking performance of open-source LLMs across benchmarks."},
    ],
}


def main():
    print("Initializing research cache with sample data...\n")

    for query, sources in SAMPLE_QUERIES.items():
        cache_sources(query, sources)
        print(f"  Cached: \"{query}\" ({len(sources)} sources)")

    stats = get_cache_stats()
    print(f"\nCache stats:")
    print(f"  Cached queries:  {stats['cached_queries']}")
    print(f"  Indexed sources: {stats['indexed_sources']}")
    print(f"  Total hits:      {stats['total_hits']}")
    print("\nDone. Cache is ready at data/research_cache.db")


if __name__ == "__main__":
    main()
