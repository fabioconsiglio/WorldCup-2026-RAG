"""Export a ChromaDB collection to a JSON backup file."""

from __future__ import annotations

import json
import sys

from rag_chromadb.config import WC_COLLECTION
from rag_chromadb.db import get_collection


def export_collection(
    collection_name: str = WC_COLLECTION,
    output_path: str = "database_backup.json",
) -> int:
    """Write a collection's data to a JSON file. Returns the document count."""
    print(f"Fetching data from '{collection_name}'...")
    collection = get_collection(collection_name)

    all_data = collection.get(include=["documents", "metadatas", "embeddings"])

    # Convert embeddings to plain floats for JSON serialization
    safe_embeddings: list[list[float] | None] = []
    if all_data.get("embeddings") is not None:
        print("Converting vector embeddings to JSON-safe format...")
        for vector in all_data["embeddings"]:
            if vector is not None:
                safe_embeddings.append([float(num) for num in vector])
            else:
                safe_embeddings.append(None)
    all_data["embeddings"] = safe_embeddings

    clean_data = {
        "ids": all_data.get("ids", []),
        "documents": all_data.get("documents", []),
        "metadatas": all_data.get("metadatas", []),
        "embeddings": safe_embeddings,
    }

    print(f"Writing {len(clean_data['ids'])} documents to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=4, ensure_ascii=False)

    print(f"Done — {len(clean_data['ids'])} documents backed up.")
    return len(clean_data["ids"])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a ChromaDB collection to JSON.",
    )
    parser.add_argument(
        "--collection",
        "-c",
        default=WC_COLLECTION,
        help=f"Collection name (default: {WC_COLLECTION}).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="database_backup.json",
        help="Output JSON file path (default: database_backup.json).",
    )
    args = parser.parse_args()

    try:
        export_collection(args.collection, args.output)
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
