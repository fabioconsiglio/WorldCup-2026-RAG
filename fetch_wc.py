"""Fetch World Cup match data from football-data.org and store in ChromaDB.

Supports incremental ETL: only re-processes matches that have changed since
the last successful sync (tracked via ``etl_state.json``).
"""

from __future__ import annotations

import json
import sys

import requests
from pydantic import ValidationError

from config import (
    ETL_STATE_FILE,
    FOOTBALL_DATA_API_TOKEN,
    FOOTBALL_DATA_API_URL,
    WC_COLLECTION,
)
from db import get_collection, get_embedding
from schemas import Match, MatchResponse, SyncState


# ---------------------------------------------------------------------------
# State (incremental ETL)
# ---------------------------------------------------------------------------

def load_sync_state() -> SyncState:
    """Load the last sync timestamp from disk, or return a fresh state."""
    try:
        with open(ETL_STATE_FILE) as f:
            return SyncState.model_validate(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, ValidationError):
        return SyncState()


def save_sync_state(state: SyncState) -> None:
    """Persist the sync state to disk."""
    with open(ETL_STATE_FILE, "w") as f:
        f.write(state.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# API fetch + validation
# ---------------------------------------------------------------------------

def fetch_matches() -> MatchResponse:
    """Fetch World Cup matches and validate the response with Pydantic.

    Raises SystemExit on auth or network errors.
    """
    if not FOOTBALL_DATA_API_TOKEN:
        print("Error: FOOTBALL_DATA_API_TOKEN environment variable is not set.")
        print("Get a free token at https://www.football-data.org/client/register")
        sys.exit(1)

    print("Fetching latest match data from football-data.org...")
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_TOKEN}

    try:
        response = requests.get(FOOTBALL_DATA_API_URL, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"API request failed: {exc}")
        sys.exit(1)

    raw = response.json()

    try:
        validated = MatchResponse.model_validate(raw)
    except ValidationError as exc:
        print(f"API response validation failed — schema mismatch:\n{exc}")
        sys.exit(1)

    print(f"Retrieved {len(validated.matches)} matches (validated OK).")
    return validated


# ---------------------------------------------------------------------------
# Incremental filter
# ---------------------------------------------------------------------------

def filter_changed(
    matches: list[Match],
    state: SyncState,
) -> tuple[list[Match], list[Match]]:
    """Split matches into new/changed vs unchanged.

    A match is considered changed when its ``lastUpdated`` is after the
    last sync timestamp, or when no prior sync exists.

    Returns (changed, skipped).
    """
    if state.last_sync is None:
        # First run — process everything
        return matches, []

    changed: list[Match] = []
    skipped: list[Match] = []

    for m in matches:
        if m.lastUpdated is None or m.lastUpdated > state.last_sync:
            changed.append(m)
        else:
            skipped.append(m)

    return changed, skipped


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def sync_matches(matches: list[Match]) -> int:
    """Upsert validated Match objects into ChromaDB. Returns upsert count."""
    if not matches:
        return 0

    collection = get_collection(WC_COLLECTION)

    for m in matches:
        collection.upsert(
            ids=[f"wc_match_{m.id}"],
            embeddings=[get_embedding(m.summary)],
            documents=[m.summary],
            metadatas=[m.metadata],
        )

    return len(matches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch World Cup matches and store in ChromaDB."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full refresh (ignore incremental state).",
    )
    args = parser.parse_args()

    state = SyncState() if args.full else load_sync_state()

    match_response = fetch_matches()

    changed, skipped = filter_changed(match_response.matches, state)
    print(
        f"Incremental: {len(changed)} changed, "
        f"{len(skipped)} unchanged → processing {len(changed)}."
    )

    count = sync_matches(changed)

    if changed:
        # Record the newest lastUpdated as the watermark
        new_watermark = max(
            (m.lastUpdated for m in changed if m.lastUpdated is not None),
            default=None,
        )
        state = SyncState(last_sync=new_watermark, match_count=state.match_count + count)
        save_sync_state(state)
        print(f"Sync state saved (watermark: {state.last_sync}).")

    print(f"Done — {count} matches upserted into '{WC_COLLECTION}'.")


if __name__ == "__main__":
    main()