"""Lightweight keyword re-ranking to boost domain-relevant chunks after vector search."""

from __future__ import annotations

import re

PERCENTAGE_RE = re.compile(r"\d+%")

PLACEMENT_QUERY_TERMS = ("average", "on average", "percentage", "placed into", "placement", "housing styles")
REVIEW_QUERY_TERMS = ("review", "students say", "noise", "wall", "thin", "loud", "roomsurf")
POLICY_QUERY_TERMS = ("microwave", "furniture", "bring", "prohibited", "allowed", "not permitted")


def _query_has_any(query: str, terms: tuple[str, ...]) -> bool:
    lower = query.lower()
    return any(term in lower for term in terms)


def _text_has_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def score_chunk(query: str, hit: dict) -> float:
    """Return an adjusted distance (lower is better). Negative boost improves rank."""
    boost = 0.0
    text = hit.get("text", "")
    section = (hit.get("section") or "").lower()
    text_lower = text.lower()

    if _query_has_any(query, PLACEMENT_QUERY_TERMS):
        if "housing statistics" in section or "average placement" in text_lower:
            boost -= 0.18
        if PERCENTAGE_RE.search(text) and _text_has_any(text_lower, ("apartment", "suite", "traditional")):
            boost -= 0.10
        if "timeline" in section or ("preference form due" in text_lower and "average placement" not in text_lower):
            boost += 0.12

    if _query_has_any(query, REVIEW_QUERY_TERMS):
        if _text_has_any(text_lower, ("thin wall", "thin walls", "noise", "nupd", "loud")):
            boost -= 0.08
        if hit.get("source") == "dorm_review.txt":
            boost -= 0.04

    if _query_has_any(query, POLICY_QUERY_TERMS):
        if _text_has_any(text_lower, ("microwave", "outside furniture", "prohibited items", "not permitted")):
            boost -= 0.15
        if section in {"microwave and refrigerator", "what not to bring", "prohibited items:"}:
            boost -= 0.10

    if _query_has_any(query, ("rate", "cost", "price", "semester", "$")):
        if hit.get("source") == "room_rates.txt" and section not in {"overview/notes"}:
            boost -= 0.05

    if _query_has_any(query, ("deadline", "due", "deposit", "application")):
        if hit.get("source") == "application_process.txt":
            boost -= 0.05
        if hit.get("source") == "spring_housing.txt" and "timeline" in section:
            boost += 0.08

    return hit["distance"] + boost


def rerank(query: str, hits: list[dict], k: int) -> list[dict]:
    """Re-order vector hits using lightweight domain heuristics."""
    if not hits:
        return []

    scored = []
    for hit in hits:
        adjusted = score_chunk(query, hit)
        scored.append({**hit, "adjusted_distance": adjusted})

    scored.sort(key=lambda item: item["adjusted_distance"])
    return scored[:k]
