"""
generation/qa_chain.py

Wraps retrieval + reranking + LLM generation into a single pipeline.
Returns the answer along with source citations.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import build_reranker, rerank


# ---------------------------------------------------------------------------
# System prompt — finance-tuned
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial research assistant with expertise in
SEC filings, earnings calls, and macroeconomic policy documents.

Answer the user's question using ONLY the provided context passages.
Be precise and cite specific figures, dates, and named items when available.
If the context does not contain enough information to answer, say so clearly —
do not hallucinate financial data.

Format your response with:
1. A direct answer in 2–4 sentences
2. Supporting evidence from the context (quote sparingly, paraphrase preferred)
3. Source citations: [filename, page N]
"""


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[dict] = field(default_factory=list)
    reranked_nodes: List[NodeWithScore] = field(default_factory=list)

    def __str__(self):
        src_str = "\n".join(
            f"  [{i+1}] {s['filename']} — page {s['page']} ({s['filing_type']}, {s['year']})"
            for i, s in enumerate(self.sources)
        )
        return f"Answer:\n{self.answer}\n\nSources:\n{src_str}"


# ---------------------------------------------------------------------------
# QA Pipeline
# ---------------------------------------------------------------------------

class FinRAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        top_k_rerank: int = None,
    ):
        self.retriever = retriever
        self.reranker = build_reranker(
            top_n=top_k_rerank or int(os.getenv("TOP_K_RERANK", 3))
        )
        self.llm = OpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            system_prompt=SYSTEM_PROMPT,
        )

    def query(
        self,
        question: str,
        ticker: Optional[str] = None,
        year: Optional[str] = None,
        filing_type: Optional[str] = None,
    ) -> RAGResponse:
        """
        Full RAG pipeline:
        1. Hybrid retrieval (dense + BM25 + RRF)
        2. Optional metadata filtering
        3. Cohere reranking
        4. LLM generation with context
        """

        # Step 1: Retrieve
        nodes = self.retriever.retrieve_with_filter(
            query=question,
            ticker=ticker,
            year=year,
            filing_type=filing_type,
        )

        # Step 2: Rerank
        reranked = rerank(question, nodes, self.reranker)

        # Step 3: Build context string
        context = self._format_context(reranked)

        # Step 4: Generate
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        response = self.llm.complete(prompt)

        # Step 5: Extract source metadata
        sources = [
            {
                "filename": n.node.metadata.get("filename", ""),
                "page": n.node.metadata.get("page", ""),
                "ticker": n.node.metadata.get("ticker", ""),
                "filing_type": n.node.metadata.get("filing_type", ""),
                "year": n.node.metadata.get("year", ""),
                "section": n.node.metadata.get("section", ""),
                "score": round(n.score or 0.0, 4),
            }
            for n in reranked
        ]

        return RAGResponse(
            query=question,
            answer=str(response),
            sources=sources,
            reranked_nodes=reranked,
        )

    def _format_context(self, nodes: List[NodeWithScore]) -> str:
        parts = []
        for i, nws in enumerate(nodes, start=1):
            m = nws.node.metadata
            header = (
                f"[Passage {i} | {m.get('filename','')} "
                f"p.{m.get('page','')} | "
                f"{m.get('filing_type','')} {m.get('year','')} | "
                f"section: {m.get('section','')}]"
            )
            parts.append(f"{header}\n{nws.node.text}")
        return "\n\n---\n\n".join(parts)
