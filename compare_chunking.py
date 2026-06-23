"""Benchmark recursive-only vs section-aware chunking on eval queries."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from chunking import chunk_document, recursive_chunk
from ingest import build_chunks, clean_document, load_documents
from tests.eval_cases import EVAL_QUERIES
from tests.test_helpers import contains_all, contains_all_across
from vector_store import build_index, retrieve


def _baseline_chunk_document(text: str, filename: str):
    """Strategy A: recursive-only for all non-rate/review docs."""
    if filename in ("room_rates.txt", "dorm_review.txt"):
        return chunk_document(text, filename)
    raw = recursive_chunk(text, filename)
    filtered = [c for c in raw if len(c.text.strip()) >= 50]
    prefixed = [c.with_metadata_prefix() for c in filtered]
    for i, chunk in enumerate(prefixed):
        chunk.chunk_index = i
    return prefixed


def _build_baseline_chunks():
    all_chunks = []
    for doc in load_documents():
        cleaned = clean_document(doc.raw_text)
        all_chunks.extend(_baseline_chunk_document(cleaned, doc.filename.strip()))
    return all_chunks


def _run_strategy(name: str, chunks, k: int = 5) -> dict:
    import hybrid_search
    import vector_store

    with tempfile.TemporaryDirectory() as tmp:
        chroma_dir = Path(tmp) / "chroma"
        bm25_path = Path(tmp) / "bm25.pkl"

        with patch("vector_store.CHROMA_DIR", chroma_dir), patch(
            "config.CHROMA_DIR", chroma_dir
        ), patch("config.BM25_INDEX_PATH", bm25_path), patch(
            "hybrid_search.BM25_INDEX_PATH", bm25_path
        ):
            vector_store._client = None
            vector_store._model = None
            hybrid_search._bm25 = None
            hybrid_search._corpus_chunks = None

            build_index(chunks, reset=True)
            results = {}
            for case in EVAL_QUERIES:
                hits = retrieve(case["query"], k=k, use_hybrid=False)
                top = hits[0] if hits else {}
                content_ok = any(contains_all(h["text"], case["must_contain"]) for h in hits)
                if not content_ok and hits:
                    content_ok = contains_all_across(hits, case["must_contain"])
                preferred = case.get("preferred_section")
                section_ok = (top.get("section") or "") == preferred if preferred else True
                results[case["id"]] = {
                    "source_ok": top.get("source") == case["expected_source"],
                    "section_ok": section_ok,
                    "content_ok": content_ok,
                    "top_section": top.get("section"),
                }
    return results


def main() -> None:
    prod_chunks = build_chunks(load_documents())
    base_chunks = _build_baseline_chunks()

    print(f"Strategy A (baseline recursive): {len(base_chunks)} chunks")
    print(f"Strategy B (production):       {len(prod_chunks)} chunks\n")

    base = _run_strategy("A", base_chunks)
    prod = _run_strategy("B", prod_chunks)

    print(f"{'Query':<6} {'Metric':<14} {'A — Baseline':<14} {'B — Production':<14}")
    print("-" * 52)
    for case in EVAL_QUERIES:
        qid = case["id"]
        for metric in ("source_ok", "section_ok", "content_ok"):
            print(
                f"Q{qid:<5} {metric:<14} "
                f"{'PASS' if base[qid][metric] else 'FAIL':<14} "
                f"{'PASS' if prod[qid][metric] else 'FAIL':<14}"
            )
        print()


if __name__ == "__main__":
    main()
