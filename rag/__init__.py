"""
CareRoot RAG package.

Modules:
- loader: convert scraped article text into LangChain Documents and chunks.
- embeddings: configure Gemini embeddings.
- vector_store: manage the shared persistent ChromaDB collection.
- ingestion: prepare and index newly scraped articles.
- retrieval: retrieve relevant evidence and format it as context.
- prompts: build evidence-grounded LangChain prompts.

Import directly from the relevant module, for example:
    from rag.loader import load_and_split_article
    from rag.ingestion import ingest_scraped_article
    from rag.retrieval import retrieve_context
"""

__all__ = [
    "loader",
    "embeddings",
    "vector_store",
    "ingestion",
    "retrieval",
    "prompts",
]
