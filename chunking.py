"""Chunking utilities: recursive splitter and special-case handlers for rates/reviews."""

from __future__ import annotations

import re

from models import Chunk

CHUNK_SIZE = 1600  # ~400 tokens
CHUNK_OVERLAP = 240  # ~60 tokens
MAX_CHUNK_SIZE = 2048  # ~512 tokens
MIN_CHUNK_LEN = 50

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
    else:
        raw_chunks = recursive_chunk(text, filename)

    filtered = [c for c in raw_chunks if len(c.text.strip()) >= MIN_CHUNK_LEN]
    prefixed = [c.with_metadata_prefix() for c in filtered]
    for i, chunk in enumerate(prefixed):
        chunk.chunk_index = i
    return prefixed
