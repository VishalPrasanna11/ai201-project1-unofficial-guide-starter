"""End-to-end generation tests (require GROQ_API_KEY)."""

from __future__ import annotations

import os

import pytest

from config import DECLINE_MESSAGE
from query import ask
from tests.eval_cases import FOLLOW_UP_TEST, IN_DOMAIN_TESTS, OUT_OF_DOMAIN_TEST
from tests.test_helpers import contains_all

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def _ensure_index(indexed_chunks):
    return indexed_chunks


@pytest.mark.parametrize("test", IN_DOMAIN_TESTS, ids=lambda t: t["name"])
def test_in_domain_generation(test, _ensure_index):
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_key_here":
        pytest.skip("GROQ_API_KEY not set")

    result = ask(test["query"])
    answer_lower = result["answer"].lower()

    source_ok = test["expected_source"] in result["sources"] or test["expected_source"] in result["answer"]
    assert source_ok
    assert contains_all(result["answer"], test["must_contain"])
    assert DECLINE_MESSAGE.lower() not in answer_lower


def test_out_of_domain_declines(_ensure_index):
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_key_here":
        pytest.skip("GROQ_API_KEY not set")

    result = ask(OUT_OF_DOMAIN_TEST["query"])
    assert DECLINE_MESSAGE.lower() in result["answer"].lower()
    assert result["sources"] == []


def test_conversational_follow_up(_ensure_index):
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_key_here":
        pytest.skip("GROQ_API_KEY not set")

    turn1 = ask(FOLLOW_UP_TEST["turn1"])
    history = [
        {"role": "user", "content": FOLLOW_UP_TEST["turn1"]},
        {"role": "assistant", "content": turn1["answer"]},
    ]
    turn2 = ask(FOLLOW_UP_TEST["turn2"], history=history)
    assert contains_all(turn2["answer"], FOLLOW_UP_TEST["must_contain"])
