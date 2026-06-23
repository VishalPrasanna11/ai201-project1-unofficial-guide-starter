"""BM25 keyword index and reciprocal rank fusion with semantic search."""

from __future__ import annotations

import pickle
import re

from rank_bm25 import BM25Okapi

from config import BM25_INDEX_PATH, BM25_TOP_K, RRF_K
from models import Chunk

_token_re = re.compile(r"[a-z0-9]+")

_bm25: BM25Okapi | None = None
_corpus_chunks: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    return _token_re.findall(text.lower())


def build_bm25_index(chunks: list[Chunk], *, persist: bool = True) -> None:
    """Build and optionally persist a BM25 index over chunk texts."""
    global _bm25, _corpus_chunks

    _corpus_chunks = [
        {
            "text": chunk.text,
            "source": chunk.source,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "id": f"{chunk.source}_{chunk.chunk_index}",
        }
        for chunk in chunks
    ]
    tokenized = [_tokenize(c["text"]) for c in _corpus_chunks]
    _bm25 = BM25Okapi(tokenized)

    if persist:
        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BM25_INDEX_PATH.open("wb") as f:
            pickle.dump({"bm25": _bm25, "corpus": _corpus_chunks}, f)


def _load_bm25_index() -> bool:
    """Load persisted BM25 index if available."""
    global _bm25, _corpus_chunks
    if _bm25 is not None and _corpus_chunks is not None:
        return True
    if not BM25_INDEX_PATH.exists():
        return False
    with BM25_INDEX_PATH.open("rb") as f:
        data = pickle.load(f)
    _bm25 = data["bm25"]
    _corpus_chunks = data["corpus"]
    return True


def bm25_search(query: str, k: int = BM25_TOP_K) -> list[dict]:
    """Return top-k BM25 hits with a pseudo-distance score (lower is better)."""
    if not _load_bm25_index() or _bm25 is None or _corpus_chunks is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = _bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    max_score = max(scores) if len(scores) else 1.0
    hits: list[dict] = []
    for idx in ranked_indices:
        chunk = _corpus_chunks[idx]
        raw = scores[idx]
        pseudo_distance = 1.0 - (raw / max_score) if max_score > 0 else 1.0
        hits.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "section": chunk.get("section"),
                "chunk_index": chunk["chunk_index"],
                "distance": pseudo_distance,
                "bm25_score": raw,
            }
        )
    return hits


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Fuse multiple ranked lists using reciprocal rank fusion."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    semantic_distances: dict[str, float] = {}

    for list_idx, results in enumerate(result_lists):
        for rank, hit in enumerate(results, 1):
            chunk_id = f"{hit['source']}_{hit['chunk_index']}"
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            if chunk_id not in by_id:
                by_id[chunk_id] = hit
            if list_idx == 0:
                semantic_distances[chunk_id] = hit["distance"]

    fused = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    hits: list[dict] = []
    for cid in fused[:k]:
        hit = {**by_id[cid]}
        if cid in semantic_distances:
            hit["distance"] = semantic_distances[cid]
        hits.append(hit)
    return hits
