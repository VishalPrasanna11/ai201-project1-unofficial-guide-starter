"""Shared test helpers."""

from __future__ import annotations


def contains_all(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return all(needle.lower() in lower for needle in needles)


def contains_all_across(hits: list[dict], needles: list[str]) -> bool:
    combined = " ".join(hit["text"] for hit in hits).lower()
    return all(needle.lower() in combined for needle in needles)
