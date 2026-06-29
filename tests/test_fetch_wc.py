"""Tests for fetch_wc.py ETL logic."""

from __future__ import annotations

from datetime import datetime, timezone

from fetch_wc import filter_changed
from schemas import Match, SyncState


def sample_match(match_id: int, last_updated: datetime | None) -> Match:
    return Match.model_validate({
        "id": match_id,
        "status": "FINISHED",
        "lastUpdated": last_updated,
    })


class TestIncrementalFilter:
    def test_first_run_processes_all(self) -> None:
        matches = [sample_match(1, None), sample_match(2, None)]
        state = SyncState()  # last_sync is None
        changed, skipped = filter_changed(matches, state)
        assert len(changed) == 2
        assert len(skipped) == 0

    def test_skips_unchanged_matches(self) -> None:
        t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        t1 = datetime(2026, 6, 10, tzinfo=timezone.utc)
        state = SyncState(last_sync=t1)
        matches = [
            sample_match(1, t0),  # older → skip
            sample_match(2, t1),  # equal → skip (not strictly after)
            sample_match(3, datetime(2026, 6, 15, tzinfo=timezone.utc)),  # newer
            sample_match(4, None),  # no timestamp → process (safety)
        ]
        changed, skipped = filter_changed(matches, state)
        assert len(changed) == 2
        assert {m.id for m in changed} == {3, 4}
        assert {m.id for m in skipped} == {1, 2}

    def test_all_changed(self) -> None:
        t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        state = SyncState(last_sync=t0)
        matches = [
            sample_match(1, datetime(2026, 6, 5, tzinfo=timezone.utc)),
            sample_match(2, datetime(2026, 6, 6, tzinfo=timezone.utc)),
        ]
        changed, skipped = filter_changed(matches, state)
        assert len(changed) == 2
        assert len(skipped) == 0

    def test_empty_match_list(self) -> None:
        state = SyncState()
        changed, skipped = filter_changed([], state)
        assert changed == []
        assert skipped == []