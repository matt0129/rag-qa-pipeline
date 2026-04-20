# FinRAG — Financial Document Q&A System

Welcome to my RAG project on Financial Documents! I built this to go deeper on RAG than most tutorials go; the typical implementation of chunking a PDF, embedding it, doing a cosine similarity search and pipe it to an LLM works fine on clean data but struggles with dense financial documents where exact terminology matters. This project is my attempt to build something that actually holds up in this specific type of situation.

The domain I picked is in SEC filings and earnings documents. I picked finance partly because the data is complex and unforgiving because made-up hallucinations of revenue figures are an obvious problem, and also partly because I wanted to branch out from my clinical research/product/consulting background and build something relevant to financial services.

## Architecture

```
PDFs (10-K Filings from 2025/2024 for Fiscal Years 2024/2023)
    │
    ▼
[Ingestion Layer]
  • PyMuPDF + pdfplumber (layout-aware extraction, fallback for scanned pages)
  • Sentence-aware chunking (SentenceSplitter, 512 tokens, 64 overlap)
  • Metadata: ticker, filing type, year, section headers
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

## Setup

```bash
# 1. Clone and install
git clone https://github.com/yourusername/finrag.git
cd finrag
pip install -r requirements.txt

# 2. Configure keys
cp .env.example .env
# Edit .env with your OpenAI and Cohere API keys

# 3. Add PDFs to your data/raw/ 
# Example Naming convention: JPM_10K_2024.pdf is how I did mine

# 4. Ingest and index
python main.py --ingest

# 5. Launch UI
python main.py --ui
```

## Getting SEC Filings

```bash
# SEC EDGAR full-text search — no API key needed
# I used JPMorgan's 10-K from 2025 and 2024, GS' 10-K from 2025 and Visa's 10-K from 2025.
# https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=JPM&type=10-K
```

## Query Examples

```bash
# CLI query
python main.py --query "What were JPMorgan's net revenues in FY2024?" --ticker JPM --year 2024 is an example

# Launch Gradio UI
python main.py --ui
```

## Evaluation Results

Run `python main.py --eval` to reproduce, these are my numbers from my run:

| Metric              | Score |
|---------------------|-------|
| Faithfulness        | 0.950 |
| Answer Relevancy    | 0.995 |
| Context Precision   | 0.717 |
| Context Recall      | 1.000 |

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
