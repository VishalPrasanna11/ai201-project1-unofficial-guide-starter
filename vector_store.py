"""Milestone 4: embed chunks and retrieve from ChromaDB."""

from __future__ import annotations

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from config import (
    BATCH_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    HYBRID_SEARCH_ENABLED,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    RRF_K,
)
from hybrid_search import bm25_search, build_bm25_index, reciprocal_rank_fusion
from metadata import infer_filters
from models import Chunk
from reranker import rerank

_model: SentenceTransformer | None = None
_client: chromadb.PersistentClient | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection(*, reset: bool = False) -> chromadb.Collection:
    client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except (ValueError, NotFoundError):
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(chunks: list[Chunk], reset: bool = True) -> int:
    """Embed all chunks and store them in ChromaDB and BM25 index."""
    if not chunks:
        raise ValueError("No chunks to index. Run ingest.build_chunks() first.")

    model = get_embedding_model()
    collection = get_collection(reset=reset)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for chunk in chunks:
        chunk_id = f"{chunk.source}_{chunk.chunk_index}"
        ids.append(chunk_id)
        documents.append(chunk.text)
        metadatas.append(
            {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section or "",
            }
        )

    embeddings: list[list[float]] = []
    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        embeddings.extend(model.encode(batch, show_progress_bar=False).tolist())

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    build_bm25_index(chunks, persist=True)
    print(f"Indexed {len(chunks)} chunks into {CHROMA_DIR}/")
    return len(chunks)


def _semantic_search(
    query: str,
    k: int,
    where: dict | None = None,
) -> list[dict]:
    collection = get_collection()
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    hits: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        hits.append(
            {
                "text": document,
                "source": metadata["source"],
                "section": metadata.get("section") or None,
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )
    return hits


def retrieve(
    query: str,
    k: int = DEFAULT_TOP_K,
    *,
    where: dict | None = None,
    use_hybrid: bool | None = None,
) -> list[dict]:
    """Return top-k chunks: metadata filter → hybrid search → re-rank."""
    if use_hybrid is None:
        use_hybrid = HYBRID_SEARCH_ENABLED

    candidate_k = min(
        get_collection().count(),
        max(k, k * RETRIEVAL_CANDIDATE_MULTIPLIER),
    )

    effective_where = where if where is not None else infer_filters(query)

    semantic_hits = _semantic_search(query, candidate_k, where=effective_where)
    if effective_where and len(semantic_hits) < k:
        semantic_hits = _semantic_search(query, candidate_k, where=None)

    if use_hybrid:
        keyword_hits = bm25_search(query, k=candidate_k)
        if keyword_hits:
            fused = reciprocal_rank_fusion([semantic_hits, keyword_hits], k=candidate_k, rrf_k=RRF_K)
        else:
            fused = semantic_hits
    else:
        fused = semantic_hits

    return rerank(query, fused, k)


def ensure_index() -> None:
    """Build the vector index if the collection is empty."""
    if get_collection().count() == 0:
        from ingest import build_chunks, load_documents

        build_index(build_chunks(load_documents()), reset=True)
