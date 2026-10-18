"""
CareRoot RAG - Embedding Configuration

Responsibilities:
- Configure the Gemini embedding model used by CareRoot.
- Provide one reusable LangChain embedding object for both documents and queries.

This module does NOT:
- Load or split documents.
- Create or manage a ChromaDB collection.
- Retrieve documents.
- Call the answer-generation LLM.

Important:
Use the same model and embedding configuration for indexing and retrieval.
If the embedding model or output dimensions change later, existing stored vectors
will generally need to be regenerated.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load API key settings from the project's .env file.
load_dotenv()

# Gemini embedding model selected for CareRoot.
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"


def _get_api_key() -> str:
    """
    Read the Gemini API key from the environment.

    GEMINI_API_KEY is preferred for CareRoot. GOOGLE_API_KEY is accepted
    as a fallback for compatibility with Google/LangChain conventions.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Add GEMINI_API_KEY to your .env file "
            "(or set GOOGLE_API_KEY)."
        )

    return api_key


@lru_cache(maxsize=1)
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Return a cached, reusable LangChain Gemini embedding model.

    LangChain's embedding interface exposes:
    - embed_documents(texts): embed document chunks for indexing.
    - embed_query(text): embed a user's search query.

    The integration selects the appropriate retrieval task type for document
    and query embedding when task_type is not explicitly overridden.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=_get_api_key(),
    )


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Generate vectors for a list of document chunks.

    This helper is useful for direct embedding calls. ChromaDB can instead
    receive get_embedding_model() and call embed_documents itself.
    """
    if not texts:
        return []

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every document text must be a non-empty string.")

    return get_embedding_model().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """
    Generate a vector for one user query, using the query embedding pathway.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Query text must be a non-empty string.")

    return get_embedding_model().embed_query(text)
