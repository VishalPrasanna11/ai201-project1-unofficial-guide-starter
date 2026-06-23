"""Milestone 5: grounded RAG generation with Groq."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from config import DECLINE_MESSAGE, DEFAULT_TOP_K, LLM_MODEL, LLM_TEMPERATURE, MAX_DISTANCE
from vector_store import retrieve

load_dotenv()

SYSTEM_PROMPT = """You are a Northeastern University housing assistant. Answer ONLY using the \
Retrieved Documents below. Do not use outside knowledge.

Rules:
- If the documents do not contain enough information, respond exactly:
  "I don't have enough information on that in my documents."
- Do not guess or infer beyond what the documents state.
- When citing facts, mention the source filename in parentheses,
  e.g. (source: room_rates.txt).
- For student reviews, distinguish them from official policy.
- When multiple reviews are relevant, synthesize themes across ALL retrieved reviews
  (e.g., thin walls, neighbor noise, NUPD proximity) — do not omit details present
  in the context.
- If any retrieved review mentions NUPD, include that detail in your answer."""


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your Groq API key."
        )
    return Groq(api_key=api_key)


def _format_context(chunks: list[dict]) -> str:
    blocks: list[str] = ["Retrieved Documents:"]
    for i, chunk in enumerate(chunks, 1):
        section = chunk.get("section")
        section_part = f" | section: {section}" if section else ""
        blocks.append(f"[{i}] (source: {chunk['source']}{section_part})")
        blocks.append(chunk["text"])
        blocks.append("")
    return "\n".join(blocks)


def normalize_content(content: Any) -> str:
    """Coerce Gradio/OpenAI-style message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
        return " ".join(parts)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)


def _retrieval_query(question: str, history: list[dict[str, str]] | None) -> str:
    """Augment retrieval query with prior user turn for follow-up questions."""
    if not history:
        return question
    last_user = next(
        (normalize_content(msg["content"]) for msg in reversed(history) if msg.get("role") == "user"),
        None,
    )
    if last_user and last_user.strip().lower() != question.strip().lower():
        return f"{last_user} {question}"
    return question


def ask(
    question: str,
    k: int = DEFAULT_TOP_K,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Retrieve relevant chunks and generate a grounded answer."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    retrieval_q = _retrieval_query(question, history)
    chunks = retrieve(retrieval_q, k=k)
    chunks = [c for c in chunks if c["distance"] < MAX_DISTANCE]

    if not chunks:
        return {
            "answer": DECLINE_MESSAGE,
            "sources": [],
            "source_details": [],
            "chunks": [],
        }

    user_message = f"{_format_context(chunks)}Question: {question}"

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history:
            messages.append(
                {"role": msg["role"], "content": normalize_content(msg["content"])}
            )
    messages.append({"role": "user", "content": user_message})

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE,
    )
    answer = response.choices[0].message.content or DECLINE_MESSAGE

    if DECLINE_MESSAGE.lower() in answer.lower():
        return {
            "answer": DECLINE_MESSAGE,
            "sources": [],
            "source_details": [],
            "chunks": [],
        }

    sources = sorted({c["source"] for c in chunks})
    source_details = [
        f"{c['source']}" + (f" — {c['section']}" if c.get("section") else "")
        for c in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "source_details": source_details,
        "chunks": chunks,
    }
