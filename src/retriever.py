import logging
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from src.ingest import get_embeddings

logger = logging.getLogger(__name__)


# Load Persisted Vectorstore 
def load_vectorstore(
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> Chroma:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {CHROMA_DIR}. "
            "Run ingestion first: python -m src.ingest"
        )

    if embeddings is None:
        embeddings = get_embeddings()

    logger.info("Loading vectorstore from %s", CHROMA_DIR)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


# Retrieval

def retrieve(query: str, vectorstore: Chroma) -> list[dict]:
    logger.info("Retrieving top-%d chunks for query: %r", TOP_K, query[:80])
    results = vectorstore.similarity_search_with_relevance_scores(query=query, k=TOP_K,)
    filtered = []
    for doc, score in results:
        logger.debug("  chunk score=%.4f  source=%s", score, doc.metadata.get("source", "?"))
        if score >= SIMILARITY_THRESHOLD:
            filtered.append(
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": round(score, 4),
                }
            )

    if not filtered:
        logger.warning(
            "No chunks passed similarity threshold (%.2f) for query: %r",
            SIMILARITY_THRESHOLD, query[:80],
        )
    else:
        logger.info("%d / %d chunks passed threshold", len(filtered), len(results))

    return filtered
