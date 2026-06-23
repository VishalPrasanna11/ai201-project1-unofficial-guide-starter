"""Gradio web UI for the NU Housing Unofficial Guide."""

import os

import gradio as gr

from query import ask, normalize_content
from vector_store import ensure_index

ensure_index()

EXAMPLE_QUESTIONS = [
    "What is the per-semester rate for a standard double room at Kerr Hall for 2025-2026?",
    "When is the housing application due for students entering in Fall 2026?",
    "What do RoomSurf students say about noise and wall thickness at International Village?",
    "On average, what housing styles are NUin spring returners placed into?",
    "Can students bring their own microwave or outside furniture to traditional or suite-style dorms?",
]


def _history_to_messages(history: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": turn["role"], "content": normalize_content(turn["content"])}
        for turn in history
    ]


def chat_fn(message: str | list | dict, history: list[dict]) -> str:
    message = normalize_content(message).strip()
    if not message:
        return ""

    try:
        prior = _history_to_messages(history) if history else None
        result = ask(message, history=prior)
        answer = result["answer"]
        if result["sources"]:
            sources = "\n".join(f"• {s}" for s in result["sources"])
            answer = f"{answer}\n\n**Retrieved from:**\n{sources}"
        return answer
    except RuntimeError as exc:
        return str(exc)


demo = gr.ChatInterface(
    fn=chat_fn,
    examples=EXAMPLE_QUESTIONS,
    title="Northeastern Housing — Unofficial Guide",
    description=(
        "Ask questions about Northeastern housing policies, rates, move-in, and student dorm reviews. "
        "Answers are grounded in retrieved documents with source attribution. "
        "Follow-up questions use conversation context."
    ),
    chatbot=gr.Chatbot(height=450, label="Conversation"),
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
