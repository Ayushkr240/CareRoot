"""
CareRoot RAG - Evidence Retrieval

Responsibilities:
- Access the shared ChromaDB vector store.
- Retrieve the most relevant stored document chunks for a question.
- Return retrieved Documents with their source metadata.
- Format retrieved evidence into a readable context block for prompts.py.

This module does NOT:
- Load PDFs or scrape webpages.
- Create or configure the embedding model.
- Ingest or index new documents.
- Call Gemini or any other LLM.
- Decide by itself whether evidence is medically sufficient.
"""

from typing import Any, Mapping, Sequence
import os
import re
from urllib.parse import urlparse

from langchain_core.documents import Document
from qdrant_client import models as qdrant_models

from .vector_store import get_vector_store


DEFAULT_TOP_K = 4
DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("CARE_ROOT_RELEVANCE_THRESHOLD", "0.55"))
DEFAULT_MIN_RELEVANT_CHUNKS = int(os.getenv("CARE_ROOT_MIN_RELEVANT_CHUNKS", "2"))
CANDIDATE_MULTIPLIER = 3
MIN_CANDIDATE_COUNT = 8


def _build_qdrant_filter(where: Mapping[str, Any] | None) -> qdrant_models.Filter | None:
    """Convert simple metadata equality filters to Qdrant's filter model.

    LangChain's Qdrant integration expects a Qdrant Filter object, and CareRoot
    stores document metadata under the nested payload key ``metadata``.
    """
    if where is None:
        return None
    if not where:
        return qdrant_models.Filter(must=[])

    conditions = [
        qdrant_models.FieldCondition(
            key=f"metadata.{key}",
            match=qdrant_models.MatchValue(value=value),
        )
        for key, value in where.items()
    ]
    return qdrant_models.Filter(must=conditions)

# CareRoot's approved medical-source policy. Subdomains are accepted.
APPROVED_SOURCE_DOMAINS = ("who.int", "cdc.gov", "nih.gov", "nice.org.uk")
EXCLUDED_SOURCE_DOMAINS = ("pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "for", "from", "give", "how", "i", "in", "is", "it",
    "me", "of", "on", "or", "please", "should", "tell", "that", "the",
    "their", "there", "this", "to", "use", "was", "what", "when", "where",
    "which", "who", "why", "with", "would", "you", "your", "about", "my",
}

# Lightweight requirement categories for deterministic question-coverage checks.
_REQUIREMENT_TERMS = {
    "symptoms": ("symptom", "symptoms", "signs", "feel", "feeling", "manifestation"),
    "causes": ("cause", "causes", "caused", "reason", "reasons", "why", "origin"),
    "treatment": ("treatment", "treat", "cure", "cures", "manage", "management", "medicine", "medication", "therapy"),
    "prevention": ("prevent", "prevention", "avoid", "protect", "protection", "reduce risk"),
    "diagnosis": ("diagnosis", "diagnose", "test", "testing", "diagnostic", "identify"),
    "risks": ("risk", "risks", "complication", "complications", "danger", "dangerous", "severe", "warning signs"),
}


def _normalise_domain(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return ""
    if "://" not in value:
        value = "https://" + value
    return (urlparse(value).hostname or "").lower().rstrip(".")


def _is_approved_source(metadata: Mapping[str, Any]) -> bool:
    url = str(metadata.get("source_url") or metadata.get("url") or "").strip()
    declared_domain = str(metadata.get("domain") or "").strip()
    host = _normalise_domain(url) or _normalise_domain(declared_domain)
    if not host:
        return False
    if any(host == excluded or host.endswith("." + excluded) for excluded in EXCLUDED_SOURCE_DOMAINS):
        return False
    return any(host == allowed or host.endswith("." + allowed) for allowed in APPROVED_SOURCE_DOMAINS)


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", (text or "").lower())
        if token not in _STOPWORDS
    }


def _question_requirements(query: str) -> list[str]:
    lowered = query.lower()
    return [name for name, terms in _REQUIREMENT_TERMS.items() if any(term in lowered for term in terms)]


def _coverage_check(query: str, documents: Sequence[Document]) -> dict[str, Any]:
    evidence = " ".join(doc.page_content for doc in documents).lower()
    requirements = _question_requirements(query)
    covered = {
        requirement: any(term in evidence for term in _REQUIREMENT_TERMS[requirement])
        for requirement in requirements
    }

    # Check the core topic separately from intent words such as symptoms/treatment.
    topic_tokens = _tokens(query)
    for terms in _REQUIREMENT_TERMS.values():
        for term in terms:
            topic_tokens -= set(_tokens(term))
    topic_covered = not topic_tokens or any(token in evidence for token in topic_tokens)
    requirements_covered = all(covered.values()) if covered else topic_covered
    return {
        "requirements": requirements,
        "covered_requirements": [key for key, value in covered.items() if value],
        "missing_requirements": [key for key, value in covered.items() if not value],
        "topic_covered": topic_covered,
        "coverage_passed": bool(requirements_covered and topic_covered),
    }


def retrieve_documents(
    query: str,
    *,
    k: int = DEFAULT_TOP_K,
    where: Mapping[str, Any] | None = None,
) -> list[Document]:
    """
    Retrieve the k most relevant chunks from the shared ChromaDB collection.

    Args:
        query: The latest healthcare question or a search-ready query.
        k: Maximum number of chunks to retrieve.
        where: Optional ChromaDB metadata filter.

    Returns:
        LangChain Documents containing page_content and source metadata.

    Note:
        Similarity search returns the nearest available chunks. It does not
        guarantee that the evidence is sufficient or clinically authoritative;
        that decision belongs to the surrounding LangGraph workflow.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    store = get_vector_store()

    if where is not None:
        return store.similarity_search(
            query=query.strip(), k=k, filter=_build_qdrant_filter(where)
        )

    return store.similarity_search(query=query.strip(), k=k)



def retrieve_scored_context(
    query: str,
    *,
    k: int = DEFAULT_TOP_K,
    where: Mapping[str, Any] | None = None,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    min_relevant_chunks: int = DEFAULT_MIN_RELEVANT_CHUNKS,
) -> dict[str, Any]:
    """Retrieve, score, and run deterministic evidence-sufficiency checks.

    This is a lightweight pre-Tavily gate. It is intentionally conservative:
    chunks must pass relevance and approved-source checks, enough chunks must
    remain, and recognized question requirements must appear in the evidence.
    It does not use an LLM or a cross-encoder reranker.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be between 0 and 1.")
    if min_relevant_chunks <= 0:
        raise ValueError("min_relevant_chunks must be greater than zero.")

    store = get_vector_store()
    candidate_k = max(k * CANDIDATE_MULTIPLIER, MIN_CANDIDATE_COUNT)
    search_kwargs: dict[str, Any] = {"query": query.strip(), "k": candidate_k}
    if where is not None:
        search_kwargs["filter"] = _build_qdrant_filter(where)

    # LangChain returns normalized relevance scores where higher is better.
    scored_results = store.similarity_search_with_relevance_scores(**search_kwargs)

    candidates: list[dict[str, Any]] = []
    accepted: list[tuple[Document, float]] = []
    for document, score_value in scored_results:
        score = float(score_value)
        metadata = document.metadata or {}
        approved_source = _is_approved_source(metadata)
        passes_threshold = score >= relevance_threshold
        candidates.append({
            "document": document,
            "score": score,
            "passes_threshold": passes_threshold,
            "approved_source": approved_source,
        })
        if passes_threshold and approved_source:
            accepted.append((document, score))

    # Keep the highest-scoring accepted chunks; no separate reranker is used yet.
    accepted.sort(key=lambda item: item[1], reverse=True)
    accepted = accepted[:k]
    documents = [item[0] for item in accepted]
    scores = [item[1] for item in accepted]
    coverage = _coverage_check(query, documents)

    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for document in documents:
        metadata = document.metadata or {}
        source_url = str(metadata.get("source_url") or "").strip()
        if source_url and source_url not in seen_urls:
            seen_urls.add(source_url)
            sources.append({
                "title": str(metadata.get("title") or "Untitled source"),
                "source_url": source_url,
                "domain": str(metadata.get("domain") or _normalise_domain(source_url)),
            })

    relevant_count = len(documents)
    count_passed = relevant_count >= min_relevant_chunks
    sufficient = count_passed and coverage["coverage_passed"]
    return {
        "query": query,
        "documents": documents,
        "scores": scores,
        "context": format_documents_as_context(documents),
        "sources": sources,
        "candidate_count": len(scored_results),
        "relevant_chunk_count": relevant_count,
        "relevance_threshold": relevance_threshold,
        "minimum_relevant_chunks": min_relevant_chunks,
        "count_passed": count_passed,
        "coverage": coverage,
        "sufficient": sufficient,
        "candidate_diagnostics": [
            {
                "score": item["score"],
                "passes_threshold": item["passes_threshold"],
                "approved_source": item["approved_source"],
                "title": str((item["document"].metadata or {}).get("title") or "Untitled source"),
                "source_url": str((item["document"].metadata or {}).get("source_url") or ""),
            }
            for item in candidates
        ],
    }


def format_documents_as_context(
    documents: Sequence[Document],
) -> str:
    """
    Format retrieved chunks into a context block for prompts.py.

    Source URLs and titles are included to support evidence traceability.
    The function formats the evidence only; it does not generate an answer.
    """
    if not documents:
        return "No relevant source documents were retrieved."

    context_parts: list[str] = []

    for index, document in enumerate(documents, start=1):
        if not isinstance(document, Document):
            raise TypeError("Every item must be a LangChain Document.")

        metadata = document.metadata or {}
        title = str(metadata.get("title") or "Untitled source")
        source_url = str(metadata.get("source_url") or "Source URL unavailable")
        domain = str(metadata.get("domain") or "")
        chunk_index = metadata.get("chunk_index")

        source_details = [
            f"Title: {title}",
            f"Source URL: {source_url}",
        ]
        if domain:
            source_details.append(f"Domain: {domain}")
        if chunk_index is not None:
            source_details.append(f"Chunk index: {chunk_index}")

        context_parts.append(
            f"[Evidence {index}]\n"
            + "\n".join(source_details)
            + "\nContent:\n"
            + document.page_content.strip()
        )

    return "\n\n---\n\n".join(context_parts)


def retrieve_context(
    query: str,
    *,
    k: int = DEFAULT_TOP_K,
    where: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convenience function to retrieve evidence and format it for a prompt.

    Returns a dictionary containing:
    - query: the submitted query
    - documents: the retrieved LangChain Documents
    - context: formatted evidence text
    - sources: unique source metadata records

    The returned context can be passed to prompts.py. The documents and source
    records are also available for citation handling and debugging.
    """
    documents = retrieve_documents(query=query, k=k, where=where)

    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for document in documents:
        metadata = document.metadata or {}
        source_url = str(metadata.get("source_url") or "").strip()

        if source_url and source_url not in seen_urls:
            seen_urls.add(source_url)
            sources.append(
                {
                    "title": str(metadata.get("title") or "Untitled source"),
                    "source_url": source_url,
                    "domain": str(metadata.get("domain") or ""),
                }
            )

    return {
        "query": query,
        "documents": documents,
        "context": format_documents_as_context(documents),
        "sources": sources,
    }
