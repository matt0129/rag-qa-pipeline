"""
retrieval/retriever.py

Hybrid retrieval: dense vector search (OpenAI embeddings + Qdrant)
combined with sparse BM25, fused via Reciprocal Rank Fusion (RRF).

Why hybrid?
- Dense retrieval is great for semantic similarity
- BM25 is great for exact keyword/ticker/term matches
  (e.g., "LIBOR transition" won't always have a good semantic neighbor)
- RRF combines both without needing to tune score weights
"""

import os
from typing import List, Optional

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(nodes: List[TextNode]) -> VectorStoreIndex:
    """
    Embed nodes and store in Qdrant.
    Supports both local (in-memory) and cloud Qdrant.
    """
    mode = os.getenv("QDRANT_MODE", "local")
    collection = os.getenv("COLLECTION_NAME", "finrag_docs")
    embed_model_name = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # Connect to Qdrant
    if mode == "cloud":
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
    else:
        client = QdrantClient(path="./data/qdrant_local")   # persists to disk

    # Create collection if it doesn't exist
    _ensure_collection(client, collection, embed_dim=1536)

    embed_model = OpenAIEmbedding(model=embed_model_name)
    vector_store = QdrantVectorStore(client=client, collection_name=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(f"Index built: {len(nodes)} nodes → Qdrant collection '{collection}'")
    return index


def _ensure_collection(client: QdrantClient, name: str, embed_dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )


# ---------------------------------------------------------------------------
# Hybrid retriever (dense + BM25 + RRF)
# ---------------------------------------------------------------------------

class HybridRetriever(BaseRetriever):
    """
    Combines a dense vector retriever and a BM25 retriever using
    Reciprocal Rank Fusion (RRF).

    RRF score: sum(1 / (k + rank_i)) across retrievers
    k=60 is the standard default (from the original RRF paper).
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        nodes: List[TextNode],
        top_k: int = None,
        rrf_k: int = 60,
        filters: Optional[dict] = None,
    ):
        self.top_k = top_k or int(os.getenv("TOP_K_RETRIEVAL", 10))
        self.rrf_k = rrf_k
        self.filters = filters

        # Dense retriever
        self.dense_retriever = index.as_retriever(similarity_top_k=self.top_k)

        # Sparse BM25 retriever (operates on the same node corpus)
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self.top_k,
        )

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str

        dense_results = self.dense_retriever.retrieve(query_str)
        bm25_results = self.bm25_retriever.retrieve(query_str)

        return self._reciprocal_rank_fusion(dense_results, bm25_results)

    def _reciprocal_rank_fusion(
        self,
        *result_lists: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """Merge multiple ranked lists using RRF."""
        scores: dict[str, float] = {}
        nodes: dict[str, NodeWithScore] = {}

        for result_list in result_lists:
            for rank, nws in enumerate(result_list, start=1):
                node_id = nws.node.node_id
                scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (self.rrf_k + rank)
                nodes[node_id] = nws

        # Re-rank by fused score
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            NodeWithScore(node=nodes[nid].node, score=score)
            for nid, score in fused[: self.top_k]
        ]

    def retrieve_with_filter(
        self,
        query: str,
        ticker: Optional[str] = None,
        year: Optional[str] = None,
        filing_type: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """
        Retrieve with optional metadata pre-filters.
        Filters applied before fusion to reduce noise.
        """
        results = self.retrieve(query)

        if ticker:
            results = [r for r in results if r.node.metadata.get("ticker") == ticker]
        if year:
            results = [r for r in results if r.node.metadata.get("year") == year]
        if filing_type:
            results = [r for r in results if r.node.metadata.get("filing_type") == filing_type]

        return results
