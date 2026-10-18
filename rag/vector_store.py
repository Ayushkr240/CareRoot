"""CareRoot RAG vector-store adapter for Qdrant Cloud.

Keeps the helper interface used by CareRoot's ingestion/retrieval modules.
The existing local chroma_db folder is not modified or deleted.
"""
import hashlib
import os
from functools import lru_cache
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from .embeddings import get_embedding_model

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("CARE_ROOT_QDRANT_COLLECTION", "careroot_medical_knowledge")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return a cached authenticated Qdrant Cloud client."""
    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL is missing from .env.")
    if not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_API_KEY is missing from .env.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    """Return the shared Qdrant collection, creating it if necessary."""
    client = get_qdrant_client()
    embedding_model = get_embedding_model()
    if not client.collection_exists(COLLECTION_NAME):
        # Determine the real dimension from the configured embedding model.
        vector_size = len(embedding_model.embed_query("CareRoot dimension check"))
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
        validate_embeddings=True,
        validate_collection_config=True,
    )


def make_chunk_id(document: Document, fallback_index: int | None = None) -> str:
    """Create a stable Qdrant-compatible UUID from source, index, and content."""
    source_url = str(document.metadata.get("source_url", "")).strip()
    if not source_url:
        raise ValueError("Every document chunk must contain metadata['source_url'] before it can be indexed.")
    chunk_index = document.metadata.get("chunk_index", fallback_index)
    payload = f"{source_url}\n{chunk_index}\n{document.page_content}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def add_documents(documents: Iterable[Document], *, batch_size: int = 100) -> list[str]:
    """Upsert chunks with stable IDs and return IDs in input order (deduplicated)."""
    docs = list(documents)
    if not docs:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    for document in docs:
        if not isinstance(document, Document):
            raise TypeError("Every item must be a LangChain Document.")
        if not document.page_content.strip():
            raise ValueError("Document chunks cannot have empty page_content.")
    ids = [make_chunk_id(doc, fallback_index=i) for i, doc in enumerate(docs)]
    unique: dict[str, Document] = {}
    for point_id, doc in zip(ids, docs):
        unique.setdefault(point_id, doc)
    unique_ids, unique_docs = list(unique), list(unique.values())
    store = get_vector_store()
    for start in range(0, len(unique_docs), batch_size):
        store.add_documents(documents=unique_docs[start:start + batch_size], ids=unique_ids[start:start + batch_size])
    return unique_ids


def get_collection_count() -> int:
    """Return the number of points in the Qdrant collection."""
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return 0
    return client.count(collection_name=COLLECTION_NAME, exact=True).count


def clear_collection() -> None:
    """Delete all points but keep the collection; use only for deliberate cleanup."""
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME, limit=256, offset=offset,
            with_payload=False, with_vectors=False,
        )
        if points:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=[point.id for point in points]),
                wait=True,
            )
        if next_offset is None:
            break
        offset = next_offset


def delete_source(source_url: str) -> int:
    """Delete chunks whose nested metadata.source_url exactly matches the URL."""
    if not source_url or not source_url.strip():
        raise ValueError("source_url cannot be empty.")
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return 0
    source_filter = models.Filter(must=[models.FieldCondition(
        key="metadata.source_url", match=models.MatchValue(value=source_url.strip())
    )])
    count = client.count(collection_name=COLLECTION_NAME, count_filter=source_filter, exact=True).count
    if count:
        client.delete(collection_name=COLLECTION_NAME, points_selector=models.FilterSelector(filter=source_filter), wait=True)
    return count


def close_vector_store() -> None:
    """Close cached Qdrant client and clear cached instances where supported."""
    client = get_qdrant_client()
    try:
        client.close()
    finally:
        get_vector_store.cache_clear()
        get_qdrant_client.cache_clear()
