"""Milestone 4: test retrieval with evaluation queries from planning.md."""

from __future__ import annotations

from ingest import build_chunks, load_documents
from vector_store import build_index, retrieve

EVAL_QUERIES = [
    {
        "id": 1,
        "query": "What is the per-semester rate for a standard double room at Kerr Hall for 2025-2026?",
        "expected_source": "room_rates.txt",
        "must_contain": ["$5,315", "Kerr Hall"],
    },
    {
        "id": 2,
        "query": (
            "When is the housing application due for students entering in Fall 2026, "
            "and when must the enrollment deposit be paid?"
        ),
        "expected_source": "application_process.txt",
        "must_contain": ["May 7, 2026", "May 1, 2026"],
    },
    {
        "id": 3,
        "query": "What do RoomSurf students say about noise and wall thickness at International Village?",
        "expected_source": "dorm_review.txt",
        "must_contain": ["thin", "wall"],
    },
    {
        "id": 4,
        "query": "On average, what housing styles are NUin spring returners placed into?",
        "expected_source": "spring_housing.txt",
        "must_contain": ["85%"],
    },
    {
        "id": 5,
        "query": (
            "Can students bring their own microwave or outside furniture "
            "to traditional or suite-style dorms?"
        ),
        "expected_source": "what_to_bring.txt",
        "must_contain": ["microwave", "outside furniture"],
    },
]


def _contains_all(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return all(needle.lower() in lower for needle in needles)


def run_eval_queries(k: int = 5) -> bool:
    """Run all evaluation queries and print results. Returns True if all pass."""
    all_passed = True

    for item in EVAL_QUERIES:
        query = item["query"]
        print(f"\n{'=' * 70}")
        print(f"Query {item['id']}: {query}")
        print(f"{'=' * 70}")

        results = retrieve(query, k=k)
        if not results:
            print("FAIL: No results returned.")
            all_passed = False
            continue

        for rank, hit in enumerate(results, 1):
            preview = hit["text"][:300].replace("\n", " ")
            if len(hit["text"]) > 300:
                preview += "..."
            section = hit["section"] or "N/A"
            print(f"\n--- Result {rank} (distance: {hit['distance']:.4f}) ---")
            print(
                f"Source: {hit['source']} | Section: {section} | Index: {hit['chunk_index']}"
            )
            print(preview)

        top = results[0]
        source_ok = top["source"] == item["expected_source"]
        distance_ok = top["distance"] < 0.5
        content_ok = any(_contains_all(hit["text"], item["must_contain"]) for hit in results)

        print(f"\nSummary for query {item['id']}:")
        print(f"  Expected source ({item['expected_source']}): {'PASS' if source_ok else 'FAIL'}")
        print(f"  Top distance < 0.5 ({top['distance']:.4f}): {'PASS' if distance_ok else 'FAIL'}")
        print(f"  Key content in top-{k} results: {'PASS' if content_ok else 'FAIL'}")

        if not (source_ok and distance_ok and content_ok):
            all_passed = False

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 70}")
    return all_passed


def main() -> None:
    print("Building chunks from documents...")
    chunks = build_chunks(load_documents())
    print(f"Total chunks: {len(chunks)}")

    print("\nEmbedding and indexing...")
    build_index(chunks, reset=True)

    run_eval_queries(k=5)


if __name__ == "__main__":
    main()
