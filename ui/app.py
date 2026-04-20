"""
ui/app.py

Gradio interface for FinRAG.
Run: python ui/app.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Lazy-import pipeline to avoid errors if not yet initialized
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        raise RuntimeError(
            "Pipeline not initialized. Run `python main.py --ingest` first "
            "to build the index, then restart the UI."
        )
    return _pipeline


def answer_question(
    question: str,
    ticker_filter: str,
    year_filter: str,
    filing_filter: str,
) -> tuple[str, str]:
    """Main handler for the Gradio interface."""
    if not question.strip():
        return "Please enter a question.", ""

    pipe = get_pipeline()

    resp = pipe.query(
        question=question,
        ticker=ticker_filter or None,
        year=year_filter or None,
        filing_type=filing_filter or None,
    )

    sources_md = "\n".join(
        f"**[{i+1}]** `{s['filename']}` — page {s['page']} "
        f"| {s['filing_type']} {s['year']} "
        f"| section: _{s['section']}_ "
        f"| rerank score: {s['score']}"
        for i, s in enumerate(resp.sources)
    )

    return resp.answer, sources_md


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="FinRAG") as demo:
    gr.Markdown(
        """
        # 📈 FinRAG — Financial Document Q&A
        Ask questions over SEC filings, earnings call transcripts, and FOMC minutes.
        Powered by hybrid retrieval (dense + BM25), Cohere reranking, and GPT-4o-mini.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. What were JPMorgan's net revenues in 2023?",
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("**Optional filters**")
            ticker_filter = gr.Dropdown(
                label="Ticker",
                choices=["", "JPM", "V", "GS", "MS", "BAC"],
                value="",
            )
            year_filter = gr.Dropdown(
                label="Year",
                choices=["", "2023", "2024", "2025"],
                value="",
            )
            filing_filter = gr.Dropdown(
                label="Filing type",
                choices=["", "10-K", "10-Q", "earnings_transcript", "fomc_minutes"],
                value="",
            )

    with gr.Row():
        answer_output = gr.Textbox(label="Answer", lines=6, interactive=False)

    with gr.Row():
        sources_output = gr.Markdown(label="Sources")

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, ticker_filter, year_filter, filing_filter],
        outputs=[answer_output, sources_output],
    )

    gr.Examples(
        examples=[
            ["What were JPMorgan's total net revenues for FY2023?", "JPM", "2023", "10-K"],
            ["How does Visa describe its competitive moat?", "V", "", "10-K"],
            ["What did the Fed say about inflation risks?", "", "", "fomc_minutes"],
            ["What risk factors relate to rising interest rates?", "", "", ""],
        ],
        inputs=[question_input, ticker_filter, year_filter, filing_filter],
    )


if __name__ == "__main__":
    demo.launch(share=False)
