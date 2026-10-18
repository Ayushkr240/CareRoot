"""
CareRoot RAG - Document Ingestion

Responsibilities:
- Accept a successfully scraped article (text + metadata).
- Convert it to a LangChain Document and split it into chunks.
- Add the chunks to the shared ChromaDB collection.
- Return a concise ingestion report.

This module does NOT:
- Search Tavily or scrape websites.
- Call the answer-generation LLM.
- Decide whether retrieved evidence is sufficient; LangGraph owns that decision.
"""

from typing import Any, Mapping

from langchain_core.documents import Document

from .loader import load_and_split_article
from .vector_store import add_documents


def ingest_article(
    article_text: str,
    article_metadata: Mapping[str, Any],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> dict[str, Any]:
    """
    Prepare and index one article in the shared ChromaDB collection.

    Args:
        article_text: Cleaned text from the existing scraper's `article["text"]`.
        article_metadata: Metadata from the existing scraper's `article["metadata"]`.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A report containing source information, chunk count, and stored IDs.

    Raises:
        ValueError/TypeError: If article input is invalid.
        RuntimeError: If the indexing step fails.
    """
    if not isinstance(article_text, str):
        raise TypeError("article_text must be a string.")

    if not article_text.strip():
        raise ValueError("article_text cannot be empty.")

    if not isinstance(article_metadata, Mapping):
        raise TypeError("article_metadata must be a mapping/dictionary.")

    metadata = dict(article_metadata)
    source_url = str(metadata.get("source_url", "")).strip()
    if not source_url:
        raise ValueError("article_metadata must contain a source_url.")

    # Stage 1: Convert scraped text into a LangChain Document and split it.
    chunks = load_and_split_article(
        article_text=article_text,
        article_metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError("The article produced no chunks; nothing was indexed.")

    # Stage 2: Store chunks in the shared ChromaDB collection.
    try:
        ids = add_documents(chunks)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to index article from {source_url}: {exc}"
        ) from exc

    return {
        "source_url": source_url,
        "title": str(metadata.get("title", "")),
        "chunks_created": len(chunks),
        "chunks_indexed": len(ids),
        "chunk_ids": ids,
    }


def ingest_scraped_article(article: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convenience adapter for the scraper's return format:
        {"text": "...", "metadata": {...}}

    Pass the dictionary returned by scrape_url() directly to this function.
    """
    if not isinstance(article, Mapping):
        raise TypeError("article must be a mapping/dictionary.")

    article_text = article.get("text")
    article_metadata = article.get("metadata")

    if not isinstance(article_text, str):
        raise ValueError("article must contain a string field named 'text'.")

    if not isinstance(article_metadata, Mapping):
        raise ValueError("article must contain a dictionary field named 'metadata'.")

    return ingest_article(
        article_text=article_text,
        article_metadata=article_metadata,
    )
