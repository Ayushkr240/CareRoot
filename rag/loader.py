"""
CareRoot RAG - Document Loader

Responsibilities:
1. Accept extracted article text and metadata from the existing scraper.
2. Convert the article into a LangChain Document.
3. Split the document into smaller chunks.
4. Preserve source metadata on every chunk.

This module does NOT:
- Search Tavily or scrape websites.
- Generate embeddings.
- Connect to or write to ChromaDB.
- Call the LLM.
"""

from typing import Any, Mapping, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default chunking settings.
# We can tune these later after testing retrieval quality.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def article_to_document(
    article_text: str,
    article_metadata: Mapping[str, Any],
) -> Document:
    """
    Convert one successfully scraped article into a LangChain Document.

    Args:
        article_text: Extracted, cleaned text returned by the scraper.
        article_metadata: Metadata dictionary returned by the scraper,
            such as source_url, title, domain, language, and document_type.

    Returns:
        A LangChain Document containing the article text and metadata.

    Raises:
        TypeError: If the text or metadata has an invalid type.
        ValueError: If the text is empty or metadata is missing.
    """
    if not isinstance(article_text, str):
        raise TypeError("article_text must be a string.")

    if not article_text.strip():
        raise ValueError("article_text cannot be empty.")

    if not isinstance(article_metadata, Mapping):
        raise TypeError("article_metadata must be a mapping/dictionary.")

    metadata = dict(article_metadata)

    if not metadata.get("source_url"):
        raise ValueError("article_metadata must contain a source_url.")

    # Document stores the text in page_content and source details in metadata.
    return Document(
        page_content=article_text.strip(),
        metadata=metadata,
    )


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split one or more LangChain Documents into smaller chunks.

    Each output chunk inherits its original document's metadata.
    Chunk indices are added to metadata to help identify chunks
    within a source document.

    Args:
        documents: Documents produced by article_to_document().
        chunk_size: Target maximum chunk size, measured in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of chunk Documents.

    Raises:
        ValueError: If chunking settings are invalid or no documents
            are supplied.
        TypeError: If an item is not a LangChain Document.
    """
    if not documents:
        raise ValueError("At least one Document is required.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    for document in documents:
        if not isinstance(document, Document):
            raise TypeError("Every item in documents must be a LangChain Document.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(list(documents))

    # Add a stable, source-local index for easier inspection and later indexing.
    # This is not the final ChromaDB ID; vector_store.py will own that policy.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks


def load_and_split_article(
    article_text: str,
    article_metadata: Mapping[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Convenience function: convert one scraped article to a Document
    and split it into chunks in one call.
    """
    document = article_to_document(
        article_text=article_text,
        article_metadata=article_metadata,
    )

    return split_documents(
        documents=[document],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
