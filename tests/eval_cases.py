"""Shared evaluation cases for retrieval and generation tests."""

from __future__ import annotations

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
        "preferred_section": "Housing Statistics",
    },
    {
        "id": 5,
        "query": (
            "Can students bring their own microwave or outside furniture "
            "to traditional or suite-style dorms?"
        ),
        "expected_source": "what_to_bring.txt",
        "must_contain": ["microwave", "outside furniture"],
        "preferred_section": "Microwave and Refrigerator",
    },
]

IN_DOMAIN_TESTS = [
    {
        "name": "Kerr Hall double rate",
        "query": EVAL_QUERIES[0]["query"],
        "expected_source": "room_rates.txt",
        "must_contain": ["$5,315", "room_rates"],
    },
    {
        "name": "Fall 2026 deadlines",
        "query": EVAL_QUERIES[1]["query"],
        "expected_source": "application_process.txt",
        "must_contain": ["May 7", "May 1"],
    },
    {
        "name": "International Village noise",
        "query": EVAL_QUERIES[2]["query"],
        "expected_source": "dorm_review.txt",
        "must_contain": ["thin", "wall", "nupd"],
    },
    {
        "name": "NUin spring returner placement",
        "query": EVAL_QUERIES[3]["query"],
        "expected_source": "spring_housing.txt",
        "must_contain": ["85%", "10%", "5%"],
    },
    {
        "name": "Microwave and outside furniture",
        "query": EVAL_QUERIES[4]["query"],
        "expected_source": "what_to_bring.txt",
        "must_contain": ["microwave", "outside furniture"],
    },
]

OUT_OF_DOMAIN_TEST = {
    "name": "Dining hall (not in corpus)",
    "query": "What is the best dining hall on campus?",
}

FOLLOW_UP_TEST = {
    "turn1": "What is the per-semester rate for a standard double room at Kerr Hall for 2025-2026?",
    "turn2": "What about triple rooms?",
    "must_contain": ["$5,205"],
}
