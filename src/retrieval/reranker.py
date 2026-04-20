"""
retrieval/reranker.py

Cross-encoder reranking via Cohere's rerank API.

Why rerank?
- Vector similarity != relevance to the specific query
- Cross-encoders jointly score (query, passage) pairs — much more accurate
- We retrieve top-K broadly, then rerank to top-N for the LLM context
- Net effect: better precision with minimal latency hit
"""

import os
from typing import List

from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.cohere_rerank import CohereRerank


def build_reranker(top_n: int = None) -> CohereRerank:
    """
    Build a Cohere reranker postprocessor.

    Args:
        top_n: Number of nodes to keep after reranking.
               Defaults to TOP_K_RERANK env var (default 3).
    """
    top_n = top_n or int(os.getenv("TOP_K_RERANK", 3))
    model = os.getenv("RERANK_MODEL", "rerank-english-v3.0")

    return CohereRerank(
        api_key=os.getenv("COHERE_API_KEY"),
        top_n=top_n,
        model=model,
    )


def rerank(
    query: str,
    nodes: List[NodeWithScore],
    reranker: CohereRerank = None,
) -> List[NodeWithScore]:
    """
    Rerank a list of retrieved nodes for a given query.

    Returns the top-N nodes sorted by cross-encoder relevance score.
    """
    if reranker is None:
        reranker = build_reranker()

    from llama_index.core.schema import QueryBundle
    reranked = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query))
    return reranked
