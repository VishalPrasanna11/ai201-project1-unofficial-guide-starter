"""Milestone 5: end-to-end grounded generation tests."""

from __future__ import annotations

from query import DECLINE_MESSAGE, ask
from vector_store import ensure_index

IN_DOMAIN_TESTS = [
    {
        "name": "Kerr Hall double rate",
        "query": "What is the per-semester rate for a standard double room at Kerr Hall for 2025-2026?",
        "expected_source": "room_rates.txt",
        "must_contain": ["$5,315", "room_rates"],
    },
    {
        "name": "Fall 2026 deadlines",
        "query": (
            "When is the housing application due for students entering in Fall 2026, "
            "and when must the enrollment deposit be paid?"
        ),
        "expected_source": "application_process.txt",
        "must_contain": ["May 7", "May 1"],
    },
    {
        "name": "International Village noise",
        "query": "What do RoomSurf students say about noise and wall thickness at International Village?",
        "expected_source": "dorm_review.txt",
        "must_contain": ["thin", "wall"],
    },
    {
        "name": "NUin spring returner placement",
        "query": "On average, what housing styles are NUin spring returners placed into?",
        "expected_source": "spring_housing.txt",
        "must_contain": ["85%", "10%", "5%"],
    },
    {
        "name": "Microwave and outside furniture",
        "query": (
            "Can students bring their own microwave or outside furniture "
            "to traditional or suite-style dorms?"
        ),
        "expected_source": "what_to_bring.txt",
        "must_contain": ["microwave", "outside furniture"],
    },
]

OUT_OF_DOMAIN_TEST = {
    "name": "Dining hall (not in corpus)",
    "query": "What is the best dining hall on campus?",
}


def _contains_all(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return all(needle.lower() in lower for needle in needles)


def run_tests() -> bool:
    all_passed = True

    for test in IN_DOMAIN_TESTS:
        print(f"\n{'=' * 70}")
        print(f"IN-DOMAIN: {test['name']}")
        print(f"Query: {test['query']}")
        print(f"{'=' * 70}")

        result = ask(test["query"])
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {', '.join(result['sources']) or 'none'}")

        answer_lower = result["answer"].lower()
        source_ok = test["expected_source"] in result["sources"] or test["expected_source"] in result["answer"]
        content_ok = _contains_all(result["answer"], test["must_contain"])
        grounded_ok = DECLINE_MESSAGE.lower() not in answer_lower

        print(f"\n  Expected source present: {'PASS' if source_ok else 'FAIL'}")
        print(f"  Key facts in answer: {'PASS' if content_ok else 'FAIL'}")
        print(f"  Not a decline response: {'PASS' if grounded_ok else 'FAIL'}")

        if not (source_ok and content_ok and grounded_ok):
            all_passed = False

    print(f"\n{'=' * 70}")
    print(f"OUT-OF-DOMAIN: {OUT_OF_DOMAIN_TEST['name']}")
    print(f"Query: {OUT_OF_DOMAIN_TEST['query']}")
    print(f"{'=' * 70}")

    result = ask(OUT_OF_DOMAIN_TEST["query"])
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {', '.join(result['sources']) or 'none'}")

    decline_ok = DECLINE_MESSAGE.lower() in result["answer"].lower()
    print(f"\n  Declines to answer: {'PASS' if decline_ok else 'FAIL'}")
    if not decline_ok:
        all_passed = False

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 70}")
    return all_passed


def main() -> None:
    print("Ensuring vector index is ready...")
    ensure_index()
    run_tests()


if __name__ == "__main__":
    main()
