"""Pytest fixtures for the Housing RAG test suite."""

from __future__ import annotations

import pytest

from ingest import build_chunks, load_documents
from vector_store import build_index


@pytest.fixture(scope="session")
def indexed_chunks():
    """Build and index all document chunks once per test session."""
    chunks = build_chunks(load_documents())
    build_index(chunks, reset=True)
    return chunks
