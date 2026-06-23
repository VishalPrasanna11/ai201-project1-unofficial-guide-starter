"""Parameterized retrieval evaluation tests."""

from __future__ import annotations

import pytest

from tests.eval_cases import EVAL_QUERIES
from tests.test_helpers import contains_all, contains_all_across
from vector_store import retrieve


@pytest.mark.parametrize("case", EVAL_QUERIES, ids=lambda c: f"Q{c['id']}")
def test_retrieval_eval(case, indexed_chunks):
    results = retrieve(case["query"], k=5)
    assert results, f"Query {case['id']}: no results returned"

    top = results[0]
    assert top["source"] == case["expected_source"]
    assert top["distance"] < 0.55

    content_ok = any(contains_all(hit["text"], case["must_contain"]) for hit in results)
    if not content_ok:
        content_ok = contains_all_across(results, case["must_contain"])
    assert content_ok, f"Query {case['id']}: key content missing from top-5"

    preferred = case.get("preferred_section")
    if preferred:
        assert (top.get("section") or "") == preferred
