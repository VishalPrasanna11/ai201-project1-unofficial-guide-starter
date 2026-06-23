"""Milestone 4: test retrieval with evaluation queries from planning.md."""

from __future__ import annotations

from ingest import build_chunks, load_documents
from tests.eval_cases import EVAL_QUERIES
from tests.test_helpers import contains_all, contains_all_across
from vector_store import build_index, retrieve


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
        distance_ok = top["distance"] < 0.55
        content_ok = any(contains_all(hit["text"], item["must_contain"]) for hit in results)
        if not content_ok:
            content_ok = contains_all_across(results, item["must_contain"])
        section_ok = True
        preferred = item.get("preferred_section")
        if preferred:
            section_ok = (top.get("section") or "") == preferred

        print(f"\nSummary for query {item['id']}:")
        print(f"  Expected source ({item['expected_source']}): {'PASS' if source_ok else 'FAIL'}")
        print(f"  Top distance < 0.55 ({top['distance']:.4f}): {'PASS' if distance_ok else 'FAIL'}")
        print(f"  Key content in top-{k} results: {'PASS' if content_ok else 'FAIL'}")
        if preferred:
            print(
                f"  Preferred section ({preferred}): "
                f"{'PASS' if section_ok else 'FAIL'} (got: {top.get('section')})"
            )

        if not (source_ok and distance_ok and content_ok and section_ok):
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
