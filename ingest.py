"""Milestone 3: load, clean, chunk, and inspect housing documents."""

from __future__ import annotations

import html
import random
import re
from pathlib import Path

from chunking import chunk_document
from models import Chunk, Document

DOCUMENTS_DIR = Path(__file__).parent / "documents"
CLEANED_DIR = Path(__file__).parent / "data" / "cleaned"

HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+);")


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[Document]:
    """Load all .txt files from documents/ (handles trailing spaces in filenames)."""
    documents: list[Document] = []
    for path in sorted(documents_dir.iterdir()):
        if not path.is_file():
            continue
        if not path.name.strip().endswith(".txt"):
            continue
        raw_text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                filename=path.name.strip(),
                raw_text=raw_text,
                source_path=str(path),
            )
        )
    return documents


def clean_document(text: str) -> str:
    """Light cleaning for plain-text housing documents."""
    text = html.unescape(text)
    text = HTML_ENTITY_RE.sub(lambda m: html.unescape(m.group(0)), text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_cleaned_documents(documents: list[tuple[str, str]], output_dir: Path = CLEANED_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, cleaned in documents:
        safe_name = filename.strip()
        (output_dir / safe_name).write_text(cleaned, encoding="utf-8")


def build_chunks(documents: list[Document]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in documents:
        cleaned = clean_document(doc.raw_text)
        chunks = chunk_document(cleaned, doc.filename.strip())
        all_chunks.extend(chunks)
    return all_chunks


def inspect_chunks(chunks: list[Chunk]) -> None:
    """Print chunk count, health checks, and 5 representative samples."""
    total = len(chunks)
    print(f"\n{'=' * 60}")
    print(f"TOTAL CHUNKS: {total}")
    print(f"{'=' * 60}")

    if total < 50:
        print("WARNING: Fewer than 50 chunks — chunks may be too large.")
    elif total > 2000:
        print("WARNING: More than 2000 chunks — chunks may be too small.")
    elif total < 80 or total > 150:
        print(f"Note: {total} chunks (planning estimate was ~80–120; rate tables add many building chunks).")
    else:
        print("Chunk count is within expected range (~80–120 target from planning.md).")

    kerr_chunks = [c for c in chunks if c.section and "Kerr Hall" in c.section]
    if kerr_chunks:
        kerr = kerr_chunks[0]
        has_name = "Kerr Hall (KER)" in kerr.text
        has_rate = "$5,315" in kerr.text
        status = "PASS" if has_name and has_rate else "FAIL"
        print(f"\nKerr Hall spot-check: {status}")
        print(f"  - Building name present: {has_name}")
        print(f"  - Double rate $5,315 present: {has_rate}")
    else:
        print("\nKerr Hall spot-check: FAIL (no Kerr Hall chunk found)")

    review_chunks = [c for c in chunks if c.source == "dorm_review.txt"]
    print(f"\nDorm review chunks: {len(review_chunks)} (expect one per review)")

    samples: list[Chunk] = []

    def pick(source: str, predicate=None) -> Chunk | None:
        matches = [c for c in chunks if c.source == source]
        if predicate:
            matches = [c for c in matches if predicate(c)]
        return matches[0] if matches else None

    picks = [
        pick("application_process.txt"),
        pick("Living_Learning_Communities.txt", predicate=lambda c: "Engineering" in c.text or "Computer" in c.text),
        kerr_chunks[0] if kerr_chunks else pick("room_rates.txt", predicate=lambda c: "Kerr" in c.text),
        pick("dorm_review.txt", predicate=lambda c: "thin walls" in c.text.lower() or "thin wall" in c.text.lower()),
        pick("residential_utlilies.txt", predicate=lambda c: "NUwave" in c.text or "Laundry" in c.text),
    ]
    for chunk in picks:
        if chunk and chunk not in samples:
            samples.append(chunk)
        if len(samples) >= 5:
            break

    if len(samples) < 5:
        extras = [c for c in chunks if c not in samples]
        random.seed(42)
        samples.extend(random.sample(extras, min(5 - len(samples), len(extras))))

    print(f"\n{'=' * 60}")
    print("5 REPRESENTATIVE CHUNKS FOR INSPECTION")
    print(f"{'=' * 60}")
    for i, chunk in enumerate(samples[:5], 1):
        preview = chunk.text[:500] + ("..." if len(chunk.text) > 500 else "")
        print(f"\n--- Chunk {i} ---")
        print(f"Source:  {chunk.source}")
        print(f"Section: {chunk.section}")
        print(f"Length:  {len(chunk.text)} chars")
        print(preview)


def main() -> None:
    print("Loading documents...")
    documents = load_documents()
    total_chars = sum(len(d.raw_text) for d in documents)
    print(f"Loaded {len(documents)} files ({total_chars:,} characters total)")
    for doc in documents:
        print(f"  - {doc.filename} ({len(doc.raw_text):,} chars)")

    cleaned_pairs: list[tuple[str, str]] = []
    for doc in documents:
        cleaned = clean_document(doc.raw_text)
        cleaned_pairs.append((doc.filename.strip(), cleaned))

    save_cleaned_documents(cleaned_pairs)
    print(f"\nSaved cleaned documents to {CLEANED_DIR}/")

    sample_name, sample_text = next(
        (name, text) for name, text in cleaned_pairs if name == "what_to_bring.txt"
    )
    print(f"\n{'=' * 60}")
    print(f"CLEANED SAMPLE: {sample_name}")
    print(f"{'=' * 60}")
    print(sample_text[:1500])
    if len(sample_text) > 1500:
        print("\n... [truncated for display]")

    print("\nChunking documents...")
    all_chunks: list[Chunk] = []
    for doc in documents:
        cleaned = clean_document(doc.raw_text)
        chunks = chunk_document(cleaned, doc.filename.strip())
        print(f"  {doc.filename.strip()}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    inspect_chunks(all_chunks)


if __name__ == "__main__":
    main()
