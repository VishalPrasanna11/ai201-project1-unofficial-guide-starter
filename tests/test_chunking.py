"""Unit tests for chunking strategies."""

from __future__ import annotations

from ingest import build_chunks, load_documents


def test_kerr_hall_rates_intact():
    chunks = build_chunks(load_documents())
    kerr = next(c for c in chunks if c.section and "Kerr Hall" in c.section)
    assert "Kerr Hall (KER)" in kerr.text
    assert "$5,315" in kerr.text


def test_dorm_review_count():
    chunks = build_chunks(load_documents())
    reviews = [c for c in chunks if c.source == "dorm_review.txt" and c.section and "Review" in c.section]
    assert len(reviews) == 8


def test_spring_housing_statistics_section():
    chunks = build_chunks(load_documents())
    stats = [c for c in chunks if c.source == "spring_housing.txt" and c.section == "Housing Statistics"]
    assert len(stats) == 1
    assert "85%" in stats[0].text
    assert "Average placements" in stats[0].text


def test_what_to_bring_microwave_section():
    chunks = build_chunks(load_documents())
    micro = [c for c in chunks if c.source == "what_to_bring.txt" and c.section == "Microwave and Refrigerator"]
    assert len(micro) == 1
    assert "microwave" in micro[0].text.lower()


def test_total_chunk_count():
    chunks = build_chunks(load_documents())
    assert len(chunks) == 163
