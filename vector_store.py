"""Milestone 4: embed chunks and retrieve from ChromaDB."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from models import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "housing_chunks"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
DEFAULT_TOP_K = 5
BATCH_SIZE = 32

_model: SentenceTransformer | None = None
_client: chromadb.PersistentClient | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
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
    """Embed all chunks and store them in ChromaDB."""
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

    print(f"Indexed {len(chunks)} chunks into {CHROMA_DIR}/")
    return len(chunks)


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return top-k chunks most similar to the query."""
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("Vector store is empty. Run build_index() first.")

    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

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
