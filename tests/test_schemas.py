"""Tests for Pydantic schemas and data validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_chromadb.schemas import Match, MatchResponse, SyncState


class TestMatch:
    """Validation and computed property tests for the Match model."""

    def test_minimal_match_validates(self) -> None:
        m = Match.model_validate({"id": 1})
        assert m.id == 1
        assert m.status == "SCHEDULED"  # default

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            Match.model_validate({})

    def test_finished_match_properties(self) -> None:
        m = Match.model_validate(
            {
                "id": 42,
                "utcDate": "2026-06-15T20:00:00Z",
                "status": "FINISHED",
                "stage": "GROUP_STAGE",
                "homeTeam": {"name": "Germany"},
                "awayTeam": {"name": "Brazil"},
                "score": {"fullTime": {"home": 3, "away": 1}},
            }
        )
        assert m.is_finished is True
        assert m.winner == "Germany"
        assert m.home_score == 3
        assert m.away_score == 1
        assert m.date_str == "2026-06-15"
        assert m.stage_title == "Group Stage"
        assert "Germany 3 - 1 Brazil" in m.result_string

    def test_scheduled_match_not_finished(self) -> None:
        m = Match.model_validate(
            {
                "id": 7,
                "status": "SCHEDULED",
                "homeTeam": {"name": "Argentina"},
                "awayTeam": {"name": "France"},
            }
        )
        assert m.is_finished is False
        assert m.winner == "TBD"

    def test_summary_contains_match_info(self) -> None:
        m = Match.model_validate(
            {
                "id": 10,
                "utcDate": "2026-07-01T18:00:00Z",
                "status": "FINISHED",
                "stage": "FINAL",
                "venue": "MetLife Stadium",
                "homeTeam": {"name": "Spain"},
                "awayTeam": {"name": "England"},
                "score": {"fullTime": {"home": 2, "away": 1}},
            }
        )
        summary = m.summary
        assert "World Cup" in summary
        assert "Final" in summary
        assert "Spain" in summary
        assert "England" in summary
        assert "MetLife Stadium" in summary

    def test_metadata_keys(self) -> None:
        m = Match.model_validate(
            {
                "id": 1,
                "status": "FINISHED",
                "score": {"fullTime": {"home": 1, "away": 0}},
                "homeTeam": {"name": "A"},
                "awayTeam": {"name": "B"},
            }
        )
        meta = m.metadata
        assert meta["type"] == "soccer_match"
        assert meta["winner"] == "A"
        assert meta["team_home"] == "A"


class TestMatchResponse:
    def test_empty_matches(self) -> None:
        r = MatchResponse.model_validate({"matches": []})
        assert r.matches == []

    def test_multiple_matches(self) -> None:
        r = MatchResponse.model_validate(
            {
                "matches": [
                    {"id": 1, "status": "SCHEDULED"},
                    {
                        "id": 2,
                        "status": "FINISHED",
                        "score": {"fullTime": {"home": 2, "away": 2}},
                        "homeTeam": {"name": "X"},
                        "awayTeam": {"name": "Y"},
                    },
                ]
            }
        )
        assert len(r.matches) == 2
        assert r.matches[0].id == 1
        assert r.matches[1].winner == "Draw"

    def test_invalid_matches_array_raises(self) -> None:
        with pytest.raises(ValidationError):
            MatchResponse.model_validate({"matches": "not_a_list"})


class TestSyncState:
    def test_default_state(self) -> None:
        s = SyncState()
        assert s.last_sync is None
        assert s.match_count == 0

    def test_roundtrip_json(self) -> None:
        s = SyncState(match_count=5)
        data = s.model_dump_json()
        restored = SyncState.model_validate_json(data)
        assert restored.match_count == 5
        assert restored.last_sync is None
