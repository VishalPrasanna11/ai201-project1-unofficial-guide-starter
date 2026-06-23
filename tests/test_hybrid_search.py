"""Tests for hybrid search retrieval."""

from __future__ import annotations

from vector_store import retrieve


def test_hybrid_kerr_hall_rate(indexed_chunks):
    results = retrieve(
        "What is the per-semester rate for a standard double room at Kerr Hall for 2025-2026?",
        k=5,
        use_hybrid=True,
    )
    assert results[0]["source"] == "room_rates.txt"
    assert "$5,315" in results[0]["text"]


def test_hybrid_nuin_statistics_section(indexed_chunks):
    results = retrieve(
        "On average, what housing styles are NUin spring returners placed into?",
        k=5,
        use_hybrid=True,
    )
    assert results[0]["source"] == "spring_housing.txt"
    assert results[0].get("section") == "Housing Statistics"
