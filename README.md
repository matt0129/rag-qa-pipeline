# FinRAG — Financial Document Q&A System

A production-grade Retrieval-Augmented Generation (RAG) pipeline for querying SEC filings, earnings call transcripts, and FOMC minutes. Built as a demonstration of advanced RAG engineering patterns beyond the typical tutorial.

## Architecture

```
PDFs (10-K, 10-Q, Earnings Transcripts, FOMC Minutes)
    │
    ▼
[Ingestion Layer]
  • PyMuPDF + pdfplumber (layout-aware extraction, fallback for scanned pages)
  • Sentence-aware chunking (SentenceSplitter, 512 tokens, 64 overlap)
  • Metadata extraction: ticker, filing type, year, section headers
    │
    ▼
[Retrieval Layer]
  • Dense: OpenAI text-embedding-3-small → Qdrant vector store
  • Sparse: BM25 over same corpus
  • Fusion: Reciprocal Rank Fusion (RRF, k=60)
  • Reranking: Cohere rerank-english-v3.0 (cross-encoder, top-K → top-3)
  • Metadata filtering: ticker / year / filing type pre-filters
    │
    ▼
[Generation Layer]
  • GPT-4o-mini with finance-tuned system prompt
  • Context formatted with source headers (filename, page, section)
  • Citations returned alongside answer
    │
    ▼
[Evaluation Layer]
  • RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
    │
    ▼
[UI Layer]
  • Gradio interface with metadata filter dropdowns
```

## Key Design Decisions

**Why hybrid retrieval?**
Dense embeddings excel at semantic similarity but miss exact financial terms (e.g., "LIBOR," specific line items). BM25 catches keyword matches. RRF fuses rankings without requiring score normalization or weight tuning.

**Why reranking?**
Vector similarity scores are noisy proxies for answer relevance. A cross-encoder jointly scores (query, passage) pairs at much higher accuracy. Retrieving top-10 then reranking to top-3 gives better precision with manageable latency.

**Why sentence-aware chunking?**
Character-split chunking severs sentences mid-thought, degrading retrieval quality. `SentenceSplitter` respects sentence boundaries so each chunk is semantically coherent.

**Why metadata filtering?**
Financial Q&A is inherently scoped (JPM 2023 vs. Visa 2024). Pre-filtering by ticker/year/filing type reduces noise and keeps retrieved context on-topic.

## Setup

```bash
# 1. Clone and install
git clone https://github.com/yourusername/finrag.git
cd finrag
pip install -r requirements.txt

# 2. Configure keys
cp .env.example .env
# Edit .env with your OpenAI and Cohere API keys

# 3. Add PDFs to data/raw/
# Naming convention: JPM_10K_2023.pdf, VISA_earnings_transcript_Q4_2023.pdf

# 4. Ingest and index
python main.py --ingest

# 5. Launch UI
python main.py --ui
```

## Getting SEC Filings

```bash
# SEC EDGAR full-text search — no API key needed
# Example: JPMorgan 10-K
# https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=JPM&type=10-K

# FOMC minutes — free from federalreserve.gov
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
```

## Query Examples

```bash
# CLI query
python main.py --query "What were JPMorgan's net revenues in FY2023?" --ticker JPM --year 2023

# With filing type filter
python main.py --query "What inflation risks did the Fed highlight?" 

# Launch Gradio UI
python main.py --ui
```

## Evaluation Results

Run `python main.py --eval` to reproduce.

| Metric              | Score |
|---------------------|-------|
| Faithfulness        | 0.950 |
| Answer Relevancy    | 0.995 |
| Context Precision   | 0.717 |
| Context Recall      | 1.000 |

*Fill in after running eval with your document corpus.*

## Project Structure

```
finrag/
├── data/
│   ├── raw/              # Drop PDFs here
│   └── processed/        # Cached nodes, Qdrant local DB
├── src/
│   ├── ingestion/
│   │   ├── loader.py     # PDF parsing + metadata extraction
│   │   └── chunker.py    # Sentence-aware chunking
│   ├── retrieval/
│   │   ├── retriever.py  # Hybrid dense+BM25+RRF retriever
│   │   └── reranker.py   # Cohere cross-encoder reranking
│   ├── generation/
│   │   └── qa_chain.py   # Full RAG pipeline + response dataclass
│   └── evaluation/
│       └── eval_harness.py  # RAGAS evaluation
├── ui/
│   └── app.py            # Gradio interface
├── notebooks/            # Exploratory analysis
├── main.py               # CLI entry point
├── requirements.txt
└── .env.example
```

## Tech Stack

- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector DB**: Qdrant (local) / Qdrant Cloud
- **Sparse retrieval**: BM25 (LlamaIndex)
- **Reranking**: Cohere `rerank-english-v3.0`
- **LLM**: OpenAI `gpt-4o-mini`
- **Evaluation**: RAGAS
- **UI**: Gradio
- **PDF parsing**: PyMuPDF + pdfplumber
