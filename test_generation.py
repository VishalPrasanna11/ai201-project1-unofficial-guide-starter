"""Milestone 5: end-to-end grounded generation tests."""

from __future__ import annotations

from config import DECLINE_MESSAGE
from query import ask
from tests.eval_cases import IN_DOMAIN_TESTS, OUT_OF_DOMAIN_TEST
from tests.test_helpers import contains_all
from vector_store import ensure_index


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
        content_ok = contains_all(result["answer"], test["must_contain"])
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
