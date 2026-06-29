"""CLI demo: ingest sample documents and run a RAG query against them."""

from config import DEFAULT_COLLECTION, DEFAULT_N_RESULTS
from db import (
    generate_answer,
    get_collection,
    get_embedding,
    build_rag_prompt,
    query_collection,
)


def ingest_demo_docs(collection_name: str = DEFAULT_COLLECTION) -> None:
    """Insert example documents so the demo has something to retrieve."""
    collection = get_collection(collection_name)

    documents = [
        "Germany defeated Brazil in the 2014 World Cup semi-final.",
        "France won the 2018 World Cup, beating Croatia in the final.",
        "Argentina won the 2022 World Cup, defeating France on penalties.",
    ]

    print(f"Ingesting {len(documents)} demo documents into '{collection_name}'...")
    for i, doc in enumerate(documents):
        collection.upsert(
            ids=[f"demo_doc_{i}"],
            embeddings=[get_embedding(doc)],
            documents=[doc],
        )
    print("Done.\n")


def run_query(
    query: str,
    *,
    collection_name: str = DEFAULT_COLLECTION,
    model: str | None = None,
) -> None:
    """Full RAG pipeline: retrieve → prompt → generate."""
    collection = get_collection(collection_name)

    # 1. Retrieve
    print(f"Searching for: '{query}'")
    chunks = query_collection(collection, query, n_results=DEFAULT_N_RESULTS)

    if not chunks:
        print("No matching documents found.")
        return

    print(f"Retrieved {len(chunks)} chunk(s).")

    # 2. Build prompt
    prompt = build_rag_prompt(query, chunks)

    # 3. Generate
    print("Generating answer...")
    answer = generate_answer(prompt, model=model)
    if answer:
        print(f"\nFinal Answer:\n{answer}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local RAG pipeline demo")
    parser.add_argument(
        "--query", "-q",
        default="Tell me about Germany vs Brazil in the World Cup.",
        help="The question to ask (default: a sample question).",
    )
    parser.add_argument(
        "--collection", "-c",
        default=DEFAULT_COLLECTION,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model override (default: value from config / env).",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Re-ingest demo documents before querying.",
    )
    args = parser.parse_args()

    if args.ingest:
        ingest_demo_docs(args.collection)

    run_query(args.query, collection_name=args.collection, model=args.model)


if __name__ == "__main__":
    main()