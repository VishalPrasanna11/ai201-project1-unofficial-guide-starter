"""Central configuration for the Housing RAG pipeline."""

from dataclasses import dataclass
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
BM25_INDEX_PATH = PROJECT_ROOT / "data" / "bm25_index.pkl"

# Embedding & retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "housing_chunks"
DEFAULT_TOP_K = 5
RETRIEVAL_CANDIDATE_MULTIPLIER = 2
MAX_DISTANCE = 0.55
BATCH_SIZE = 32

# Hybrid search
HYBRID_SEARCH_ENABLED = True
BM25_TOP_K = 10
RRF_K = 60

# Generation
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
DECLINE_MESSAGE = "I don't have enough information on that in my documents."

# Chunking
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 240
MAX_CHUNK_SIZE = 2048
MIN_CHUNK_LEN = 50


@dataclass(frozen=True)
class RAGConfig:
    """Grouped tunable parameters for the RAG pipeline."""

    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    min_chunk_len: int = MIN_CHUNK_LEN
    top_k: int = DEFAULT_TOP_K
    max_distance: float = MAX_DISTANCE
    candidate_multiplier: int = RETRIEVAL_CANDIDATE_MULTIPLIER
    hybrid_enabled: bool = HYBRID_SEARCH_ENABLED
    bm25_top_k: int = BM25_TOP_K
    rrf_k: int = RRF_K
    llm_model: str = LLM_MODEL
    llm_temperature: float = LLM_TEMPERATURE
    decline_message: str = DECLINE_MESSAGE


CONFIG = RAGConfig()
