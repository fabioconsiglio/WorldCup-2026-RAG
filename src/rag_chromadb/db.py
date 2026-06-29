"""Shared database and embedding utilities.

Provides a single place for ChromaDB client creation, collection access,
embedding generation, and common RAG operations.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import chromadb
from openai import OpenAI

from rag_chromadb.config import (
    CHROMA_DB_PATH,
    DEFAULT_N_RESULTS,
    DEFAULT_TEMPERATURE,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

# ---------------------------------------------------------------------------
# Lazy-singleton clients (cached so they're created only once)
# ---------------------------------------------------------------------------


@cache
def get_openai_client() -> OpenAI:
    """Return a cached OpenAI client pointing at the local LM Studio server."""
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


@cache
def get_chroma_client() -> chromadb.PersistentClient:
    """Return a cached ChromaDB persistent client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def get_embedding(text: str, *, model: str = EMBEDDING_MODEL) -> list[float]:
    """Generate an embedding vector for the given text."""
    client = get_openai_client()
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def get_collection(name: str) -> chromadb.Collection:
    """Get or create a named ChromaDB collection.

    Uses get_or_create so it's safe to call repeatedly — existing data is
    never overwritten unless you explicitly upsert with the same IDs.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name=name)


# ---------------------------------------------------------------------------
# RAG pipeline helpers
# ---------------------------------------------------------------------------


def query_collection(
    collection: chromadb.Collection,
    query_text: str,
    *,
    n_results: int = DEFAULT_N_RESULTS,
) -> list[str]:
    """Query a collection and return the top matching document texts.

    Returns an empty list when no documents match.
    """
    query_embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    docs = results.get("documents", [[]])
    if docs and docs[0]:
        return docs[0]
    return []


def build_rag_prompt(query: str, context_chunks: list[str]) -> str:
    """Build a RAG prompt from a user query and retrieved context chunks."""
    context = "\n\n".join(context_chunks)
    return (
        f"Answer the query based ONLY on the following context.\n\n"
        f"Context:\n{context}\n\n"
        f"Query: {query}"
    )


def generate_answer(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    stream: bool = False,
) -> Any | None:
    """Send a prompt to the LLM and return the response.

    When ``stream`` is True the raw stream object is returned (caller must
    consume it).  Otherwise the text content is returned as a string.
    Returns ``None`` if something goes wrong.
    """
    from rag_chromadb.config import DEFAULT_LLM_MODEL

    llm_model = model or DEFAULT_LLM_MODEL
    client = get_openai_client()

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=stream,
        )
        if stream:
            return response
        return response.choices[0].message.content
    except Exception as exc:
        print(f"LLM call failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Upsert helper
# ---------------------------------------------------------------------------


def upsert_documents(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> None:
    """Embed and upsert a batch of documents into a collection."""
    embeddings = [get_embedding(doc) for doc in documents]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
