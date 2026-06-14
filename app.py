"""Gradio web UI for the NU Housing Unofficial Guide."""

import gradio as gr

from query import ask
from vector_store import ensure_index

ensure_index()


def handle_query(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""

    try:
        result = ask(question)
    except RuntimeError as exc:
        return str(exc), ""

    sources = "\n".join(f"• {source}" for source in result["sources"])
    return result["answer"], sources


with gr.Blocks(title="NU Housing Unofficial Guide") as demo:
    gr.Markdown("# Northeastern Housing — Unofficial Guide")
    gr.Markdown(
        "Ask questions about Northeastern housing policies, rates, move-in, and student dorm reviews."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. What is the Kerr Hall double room rate?",
    )
    btn = gr.Button("Ask")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)
    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()
