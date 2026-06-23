"""Tests for metadata filter inference and retrieval fallback."""

from __future__ import annotations

from metadata import infer_filters
from vector_store import retrieve


def test_review_query_filters_dorm_reviews():
    assert infer_filters("What do RoomSurf students say about noise?") == {"source": "dorm_review.txt"}


def test_rate_query_filters_room_rates():
    assert infer_filters("What is the per-semester rate for Kerr Hall?") == {"source": "room_rates.txt"}


def test_nuin_query_filters_spring_housing():
    assert infer_filters("On average, what housing styles are NUin spring returners placed into?") == {
        "source": "spring_housing.txt"
    }


def test_microwave_query_filters_what_to_bring():
    assert infer_filters("Can students bring their own microwave?") == {"source": "what_to_bring.txt"}


def test_generic_query_no_filter():
    assert infer_filters("Tell me about housing") is None


def test_metadata_filter_fallback(indexed_chunks):
    """When a narrow filter returns too few hits, retrieve falls back to unfiltered search."""
    results = retrieve(
        "What is the per-semester rate for Kerr Hall?",
        k=5,
        where={"source": "all_gender_housing.txt"},
    )
    assert results
    assert results[0]["source"] == "room_rates.txt"


def test_metadata_filter_fallback_when_no_matches(indexed_chunks):
    """Forced filter with no matches should fall back to unfiltered search."""
    results = retrieve(
        "What is the per-semester rate for a standard double room at Kerr Hall for 2025-2026?",
        k=5,
        where={"source": "nonexistent.txt"},
    )
    assert len(results) == 5
    assert results[0]["source"] == "room_rates.txt"

