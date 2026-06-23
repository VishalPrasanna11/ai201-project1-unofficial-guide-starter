"""Chunking utilities: recursive splitter and special-case handlers for rates/reviews."""

from __future__ import annotations

import re

from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CHUNK_SIZE, MIN_CHUNK_LEN
from models import Chunk

SPRING_HOUSING_SECTIONS = [
    "Housing Statistics",
    "Program Overview",
    "How to Apply",
    "Timeline",
    "Process",
    "Housing Types to Expect",
    "Placement Overview",
    "University Housing Styles",
    "Traditional Accommodations",
    "Suite Style Accommodations",
    "Apartment Style Accommodations",
    "Living Learning Communities (LLCs)",
    "Additional Housing Options",
    "Medical Accommodations",
    "All Gender Housing",
    "Frequently Asked Questions",
    "Can I choose my residence hall?",
    "When is move-in?",
    "When will I know my housing assignment?",
    "What should I bring when I move in?",
    "How do I reserve a Microfridge or ship to my room before move in?",
    "Contact Information",
]

WHAT_TO_BRING_SECTIONS = [
    "University-Provided Items",
    "Packing Recommendations",
    "Boston Climate",
    "What You May Want to Bring",
    "Clothing",
    "Medicine and Toiletries",
    "Bedding and Linens",
    "Room Supplies",
    "Microwave and Refrigerator",
    "What NOT to Bring",
    "Prohibited Items:",
]

SEPARATORS = ["\n\n", "\n", ". ", " "]

# Multi-line building block: "Kerr Hall (KER):" followed by "- rate" lines
BUILDING_BLOCK_RE = re.compile(
    r"^([^\n]+?\([A-Z0-9/]+\):\s*\n(?:- .+(?:\n|$))+)",
    re.MULTILINE,
)

# Single-line building entry: "768 Columbus Ave (768) - Double Bedroom: $6,315"
BUILDING_LINE_RE = re.compile(
    r"^([^\n]+?\([A-Z0-9/]+\) - .+)$",
    re.MULTILINE,
)

REVIEW_SPLIT_RE = re.compile(r"(?=Review \d+ \()")


def _split_text(text: str, separator: str) -> list[str]:
    if separator:
        splits = text.split(separator)
        return [s + separator if i < len(splits) - 1 else s for i, s in enumerate(splits) if s]
    return list(text)


def _merge_splits(splits: list[str], separator: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in splits:
        piece_len = len(piece)
        if current_len + piece_len > chunk_size and current:
            chunk = separator.join(current).strip()
            if chunk:
                chunks.append(chunk)
            overlap_buf: list[str] = []
            overlap_len = 0
            for part in reversed(current):
                if overlap_len + len(part) > chunk_overlap:
                    break
                overlap_buf.insert(0, part)
                overlap_len += len(part)
            current = overlap_buf
            current_len = sum(len(p) for p in current)

        current.append(piece)
        current_len += piece_len

    if current:
        chunk = separator.join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = separators[-1]
    for sep in separators:
        if sep == "":
            separator = sep
            break
        if sep in text:
            separator = sep
            break

    splits = _split_text(text, separator)
    if len(splits) == 1:
        if len(separators) > 1:
            return _recursive_split(text, separators[1:], chunk_size, chunk_overlap)
        # Hard split if no finer separator works
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    merged = _merge_splits(splits, separator, chunk_size, chunk_overlap)
    final: list[str] = []
    for chunk in merged:
        if len(chunk) > MAX_CHUNK_SIZE:
            final.extend(_recursive_split(chunk, separators[1:], chunk_size, chunk_overlap))
        else:
            final.append(chunk)
    return final


def recursive_chunk(text: str, source: str, section_prefix: str | None = None) -> list[Chunk]:
    """Default recursive chunking for structured policy/guide documents."""
    pieces = _recursive_split(text, SEPARATORS, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        if len(piece.strip()) < MIN_CHUNK_LEN:
            continue
        chunks.append(
            Chunk(
                text=piece.strip(),
                source=source,
                section=section_prefix,
                chunk_index=i,
            )
        )
    return chunks


def chunk_room_rates(text: str, source: str) -> list[Chunk]:
    """One chunk per building; preamble/notes via recursive splitter."""
    chunks: list[Chunk] = []
    idx = 0

    building_matches: list[tuple[int, int, str, str]] = []
    seen_spans: set[tuple[int, int]] = set()

    for pattern in (BUILDING_BLOCK_RE, BUILDING_LINE_RE):
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            content = match.group(0).strip()
            section = content.split("\n")[0].split(" - ")[0].strip().rstrip(":")
            building_matches.append((match.start(), match.end(), content, section))

    building_matches.sort(key=lambda x: x[0])

    cursor = 0
    for start, end, content, section in building_matches:
        if start > cursor:
            preamble = text[cursor:start].strip()
            if preamble and len(preamble) >= MIN_CHUNK_LEN:
                for chunk in recursive_chunk(preamble, source, section_prefix="Overview/Notes"):
                    chunk.chunk_index = idx
                    chunks.append(chunk)
                    idx += 1
        chunks.append(
            Chunk(text=content, source=source, section=section, chunk_index=idx)
        )
        idx += 1
        cursor = end

    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail and len(tail) >= MIN_CHUNK_LEN:
            for chunk in recursive_chunk(tail, source, section_prefix="Overview/Notes"):
                chunk.chunk_index = idx
                chunks.append(chunk)
                idx += 1

    return chunks


def chunk_by_known_headers(
    text: str,
    source: str,
    headers: list[str],
    *,
    preamble_section: str = "Overview",
) -> list[Chunk]:
    """Split a guide document at known section headers so each section is its own chunk."""
    chunks: list[Chunk] = []
    idx = 0

    # Build a regex that matches any known header at the start of a line.
    escaped = [re.escape(header) for header in sorted(headers, key=len, reverse=True)]
    header_pattern = re.compile(rf"^({'|'.join(escaped)})\s*$", re.MULTILINE)

    matches = list(header_pattern.finditer(text))
    if not matches:
        return recursive_chunk(text, source)

    cursor = 0
    for match in matches:
        if match.start() > cursor:
            preamble = text[cursor:match.start()].strip()
            if preamble and len(preamble) >= MIN_CHUNK_LEN:
                for chunk in recursive_chunk(preamble, source, section_prefix=preamble_section):
                    chunk.chunk_index = idx
                    chunks.append(chunk)
                    idx += 1

        section_name = match.group(1).strip()
        next_start = matches[matches.index(match) + 1].start() if matches.index(match) + 1 < len(matches) else len(text)
        body = text[match.start():next_start].strip()
        if len(body) >= MIN_CHUNK_LEN:
            chunks.append(Chunk(text=body, source=source, section=section_name, chunk_index=idx))
            idx += 1
        cursor = next_start

    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail and len(tail) >= MIN_CHUNK_LEN:
            for chunk in recursive_chunk(tail, source, section_prefix="Additional"):
                chunk.chunk_index = idx
                chunks.append(chunk)
                idx += 1

    return chunks


def chunk_spring_housing(text: str, source: str) -> list[Chunk]:
    """Section-aware chunking so Housing Statistics and Timeline stay separate."""
    return chunk_by_known_headers(text, source, SPRING_HOUSING_SECTIONS, preamble_section="Introduction")


def chunk_what_to_bring(text: str, source: str) -> list[Chunk]:
    """Section-aware chunking so microwave/prohibited-item rules are not buried."""
    return chunk_by_known_headers(text, source, WHAT_TO_BRING_SECTIONS, preamble_section="Overview")


def chunk_reviews(text: str, source: str) -> list[Chunk]:
    """One chunk per RoomSurf review; zero overlap."""
    chunks: list[Chunk] = []
    current_dorm: str | None = None
    idx = 0

    parts = REVIEW_SPLIT_RE.split(text)
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("Review "):
            header_line = part.split("\n")[0]
            section = f"{current_dorm} — {header_line}" if current_dorm else header_line
            if len(part) >= MIN_CHUNK_LEN:
                chunks.append(
                    Chunk(text=part, source=source, section=section, chunk_index=idx)
                )
                idx += 1
        elif "DORM REVIEWS" in part.upper():
            lines = [ln.strip() for ln in part.split("\n") if ln.strip()]
            for line in lines:
                if "DORM REVIEWS" in line.upper():
                    current_dorm = line.replace(" DORM REVIEWS", "").strip()
        elif len(part) >= MIN_CHUNK_LEN and "RoomSurf" not in part:
            chunks.append(
                Chunk(
                    text=part,
                    source=source,
                    section=current_dorm or "Introduction",
                    chunk_index=idx,
                )
            )
            idx += 1

    return chunks


def chunk_document(text: str, filename: str) -> list[Chunk]:
    """Route a cleaned document to the appropriate chunking strategy."""
    if filename == "room_rates.txt":
        raw_chunks = chunk_room_rates(text, filename)
    elif filename == "dorm_review.txt":
        raw_chunks = chunk_reviews(text, filename)
    elif filename == "spring_housing.txt":
        raw_chunks = chunk_spring_housing(text, filename)
    elif filename == "what_to_bring.txt":
        raw_chunks = chunk_what_to_bring(text, filename)
    else:
        raw_chunks = recursive_chunk(text, filename)

    filtered = [c for c in raw_chunks if len(c.text.strip()) >= MIN_CHUNK_LEN]
    prefixed = [c.with_metadata_prefix() for c in filtered]
    for i, chunk in enumerate(prefixed):
        chunk.chunk_index = i
    return prefixed
