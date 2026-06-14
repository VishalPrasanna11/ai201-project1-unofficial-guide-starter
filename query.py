"""Milestone 5: grounded RAG generation with Groq."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

from vector_store import retrieve

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
MAX_DISTANCE = 0.65
DECLINE_MESSAGE = "I don't have enough information on that in my documents."

SYSTEM_PROMPT = """You are a Northeastern University housing assistant. Answer ONLY using the \
Retrieved Documents below. Do not use outside knowledge.

Rules:
- If the documents do not contain enough information, respond exactly:
  "I don't have enough information on that in my documents."
- Do not guess or infer beyond what the documents state.
- When citing facts, mention the source filename in parentheses,
  e.g. (source: room_rates.txt).
- For student reviews, distinguish them from official policy."""


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


def ask(question: str, k: int = TOP_K) -> dict:
    """Retrieve relevant chunks and generate a grounded answer."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    chunks = retrieve(question, k=k)
    chunks = [c for c in chunks if c["distance"] < MAX_DISTANCE]

    if not chunks:
        return {
            "answer": DECLINE_MESSAGE,
            "sources": [],
            "source_details": [],
            "chunks": [],
        }

    user_message = f"{_format_context(chunks)}Question: {question}"

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content or DECLINE_MESSAGE

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
