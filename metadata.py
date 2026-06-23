"""Query-driven metadata filter inference for ChromaDB retrieval."""

from __future__ import annotations


def infer_filters(query: str) -> dict | None:
    """Return a ChromaDB where-filter dict, or None if no strong signal."""
    lower = query.lower()

    if any(term in lower for term in ("roomsurf", "students say", "review", "student review")):
        return {"source": "dorm_review.txt"}

    if any(term in lower for term in ("nuin", "spring returner", "n.u.in")):
        return {"source": "spring_housing.txt"}

    if any(term in lower for term in ("microwave", "prohibited", "what to bring", "what not to bring")):
        return {"source": "what_to_bring.txt"}

    if "outside furniture" in lower or (
        "bring" in lower and any(w in lower for w in ("furniture", "pack", "allowed"))
    ):
        return {"source": "what_to_bring.txt"}

    if any(term in lower for term in ("rate", "per-semester", "per semester", "cost", "price", "$")):
        return {"source": "room_rates.txt"}

    if any(term in lower for term in ("deadline", "deposit", "application due", "application")):
        if "nuin" not in lower:
            return {"source": "application_process.txt"}

    return None
