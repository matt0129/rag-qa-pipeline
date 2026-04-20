"""
main.py

CLI entry point for FinRAG.

Usage:
  python main.py --ingest          # Load PDFs, chunk, build index
  python main.py --query "..."     # Ask a single question
  python main.py --eval            # Run RAGAS evaluation
  python main.py --ui              # Launch Gradio UI
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()


def run_ingestion():
    from src.ingestion.loader import load_pdfs
    from src.ingestion.chunker import chunk_documents, print_chunk_stats
    from src.retrieval.retriever import build_index

    print("=== Step 1: Loading PDFs ===")
    docs = load_pdfs("data/raw")

    print("\n=== Step 2: Chunking ===")
    nodes = chunk_documents(docs)
    print_chunk_stats(nodes)

    print("\n=== Step 3: Building Index ===")
    index = build_index(nodes)

    # Cache nodes for BM25 (save to disk)
    import pickle
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/nodes.pkl", "wb") as f:
        pickle.dump(nodes, f)

    print("\n✓ Ingestion complete. Run --query or --ui next.")
    return index, nodes


def load_pipeline():
    import pickle
    from src.retrieval.retriever import build_index, HybridRetriever
    from src.generation.qa_chain import FinRAGPipeline
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from qdrant_client import QdrantClient

    # Load cached nodes
    with open("data/processed/nodes.pkl", "rb") as f:
        nodes = pickle.load(f)

    # Reconnect to existing Qdrant index
    client = QdrantClient(path="./data/qdrant_local")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=os.getenv("COLLECTION_NAME", "finrag_docs"),
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    retriever = HybridRetriever(index=index, nodes=nodes)
    pipeline = FinRAGPipeline(retriever=retriever)
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="FinRAG CLI")
    parser.add_argument("--ingest", action="store_true", help="Ingest PDFs and build index")
    parser.add_argument("--query", type=str, help="Ask a question")
    parser.add_argument("--ticker", type=str, help="Filter by ticker (e.g. JPM)")
    parser.add_argument("--year", type=str, help="Filter by year (e.g. 2023)")
    parser.add_argument("--eval", action="store_true", help="Run RAGAS evaluation")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI")
    args = parser.parse_args()

    if args.ingest:
        run_ingestion()

    elif args.query:
        pipeline = load_pipeline()
        resp = pipeline.query(
            question=args.query,
            ticker=args.ticker,
            year=args.year,
        )
        print(resp)

    elif args.eval:
        pipeline = load_pipeline()
        from src.evaluation.eval_harness import RAGEvaluator
        evaluator = RAGEvaluator(pipeline)
        evaluator.run()

    elif args.ui:
        import ui.app as app_module
        app_module._pipeline = load_pipeline()
        app_module.demo.launch()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
