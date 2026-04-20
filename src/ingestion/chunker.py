"""
ingestion/chunker.py

Sentence-aware chunking with configurable size/overlap.
Groups pages into chunks that respect sentence boundaries —
much cleaner than naive character splits.
"""

import os
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[TextNode]:
    """
    Split documents into overlapping chunks using sentence-aware splitting.

    Key design choices:
    - SentenceSplitter respects sentence boundaries (no mid-sentence cuts)
    - Metadata is propagated to every child chunk
    - Chunk index is appended to metadata for traceability

    Args:
        documents: Raw page-level Documents from loader.py
        chunk_size: Token count per chunk (default from env)
        chunk_overlap: Token overlap between chunks (default from env)

    Returns:
        List of TextNode objects ready for embedding + indexing
    """
    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 512))
    chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 64))

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
    )

    all_nodes: List[TextNode] = []

    for doc in documents:
        nodes = splitter.get_nodes_from_documents([doc])

        # Enrich each chunk with its position within the source page
        for i, node in enumerate(nodes):
            node.metadata.update({
                "chunk_index": i,
                "chunk_total": len(nodes),
            })
            # Preserve source document ID for traceability
            node.metadata["source_doc_id"] = doc.id_

        all_nodes.extend(nodes)

    print(f"Chunked {len(documents)} pages → {len(all_nodes)} nodes "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")

    return all_nodes


def print_chunk_stats(nodes: List[TextNode]) -> None:
    """Print a quick summary of chunk size distribution."""
    lengths = [len(n.text.split()) for n in nodes]
    print(f"\nChunk stats (word count):")
    print(f"  min:    {min(lengths)}")
    print(f"  max:    {max(lengths)}")
    print(f"  mean:   {sum(lengths) / len(lengths):.0f}")
    print(f"  total nodes: {len(nodes)}")
