"""Fetch World Cup match data from football-data.org and store in ChromaDB."""

from __future__ import annotations

import sys

import requests

from config import (
    FOOTBALL_DATA_API_TOKEN,
    FOOTBALL_DATA_API_URL,
    WC_COLLECTION,
)
from db import get_collection, get_embedding


def fetch_matches() -> list[dict]:
    """Fetch World Cup matches from the football-data.org API."""
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

    data = response.json()
    matches = data.get("matches", [])
    print(f"Successfully retrieved {len(matches)} matches.")
    return matches


def build_match_summary(match: dict) -> tuple[str, str, dict]:
    """Build a natural-language summary and metadata dict for one match.

    Returns (match_id, summary_text, metadata_dict).
    """
    match_id = str(match.get("id", ""))
    date = (match.get("utcDate") or "").split("T")[0]
    stage = (match.get("stage") or "Unknown Stage").replace("_", " ").title()
    home_team = (match.get("homeTeam") or {}).get("name", "Unknown Team")
    away_team = (match.get("awayTeam") or {}).get("name", "Unknown Team")
    venue = match.get("venue", "an unspecified stadium")

    score_data = (match.get("score") or {}).get("fullTime") or {}
    home_score = score_data.get("home")
    away_score = score_data.get("away")
    status = match.get("status")

    if status == "FINISHED" and home_score is not None and away_score is not None:
        result_string = f"The final score was {home_team} {home_score} - {away_score} {away_team}."
        if home_score > away_score:
            winner = home_team
        elif away_score > home_score:
            winner = away_team
        else:
            winner = "Draw"
    else:
        result_string = (
            f"The match is scheduled/current status is {status} "
            f"and has not finished yet."
        )
        winner = "TBD"

    summary = (
        f"In the World Cup {stage} match played on {date} at {venue}, "
        f"{home_team} faced off against {away_team}. {result_string}"
    )

    metadata = {
        "date": date,
        "stage": stage,
        "team_home": home_team,
        "team_away": away_team,
        "status": status,
        "winner": winner,
        "type": "soccer_match",
    }

    return match_id, summary, metadata


def sync_matches(matches: list[dict], collection_name: str = WC_COLLECTION) -> int:
    """Upsert all matches into ChromaDB. Returns the count of upserted matches."""
    collection = get_collection(collection_name)

    for match in matches:
        match_id, summary, metadata = build_match_summary(match)
        collection.upsert(
            ids=[f"wc_match_{match_id}"],
            embeddings=[get_embedding(summary)],
            documents=[summary],
            metadatas=[metadata],
        )

    return len(matches)


def main() -> None:
    matches = fetch_matches()
    count = sync_matches(matches)
    print(f"\nChromaDB synced — {count} matches in '{WC_COLLECTION}'.")


if __name__ == "__main__":
    main()