"""Pydantic models for data validation at API boundaries."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Football-data.org API response models
# ---------------------------------------------------------------------------


class Team(BaseModel):
    name: str = "Unknown Team"


class Score(BaseModel):
    home: int | None = None
    away: int | None = None


class FullTimeScore(BaseModel):
    fullTime: Score | None = None


class Match(BaseModel):
    """A single match from the football-data.org /matches endpoint."""

    id: int
    utcDate: str = ""
    status: str = "SCHEDULED"
    stage: str = "UNKNOWN"
    venue: str | None = "unspecified stadium"
    homeTeam: Team | None = None
    awayTeam: Team | None = None
    score: FullTimeScore | None = None
    lastUpdated: datetime | None = None

    @field_validator("utcDate", mode="before")
    @classmethod
    def coerce_date_to_str(cls, v: object) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v) if v else ""

    @property
    def date_str(self) -> str:
        """YYYY-MM-DD extracted from utcDate."""
        return (self.utcDate or "").split("T")[0]

    @property
    def stage_title(self) -> str:
        """Human-readable stage name."""
        return (self.stage or "UNKNOWN").replace("_", " ").title()

    @property
    def home_name(self) -> str:
        if self.homeTeam:
            return self.homeTeam.name
        return "Unknown Team"

    @property
    def away_name(self) -> str:
        if self.awayTeam:
            return self.awayTeam.name
        return "Unknown Team"

    @property
    def home_score(self) -> int | None:
        ft = (self.score and self.score.fullTime) or None
        return ft.home if ft else None

    @property
    def away_score(self) -> int | None:
        ft = (self.score and self.score.fullTime) or None
        return ft.away if ft else None

    @property
    def is_finished(self) -> bool:
        return (
            self.status == "FINISHED"
            and self.home_score is not None
            and self.away_score is not None
        )

    @property
    def result_string(self) -> str:
        if self.is_finished:
            return (
                f"The final score was {self.home_name} {self.home_score}"
                f" - {self.away_score} {self.away_name}."
            )
        return (
            f"The match is scheduled/current status is {self.status}"
            f" and has not finished yet."
        )

    @property
    def winner(self) -> str:
        if not self.is_finished:
            return "TBD"
        if self.home_score == self.away_score:  # type: ignore[operator]
            return "Draw"
        home = self.home_score or 0
        away = self.away_score or 0
        return self.home_name if home > away else self.away_name  # type: ignore[return-value]

    @property
    def summary(self) -> str:
        return (
            f"In the World Cup {self.stage_title} match played on "
            f"{self.date_str} at {self.venue or 'an unspecified stadium'}, "
            f"{self.home_name} faced off against {self.away_name}. "
            f"{self.result_string}"
        )

    @property
    def metadata(self) -> dict[str, str | None]:
        return {
            "date": self.date_str,
            "stage": self.stage_title,
            "team_home": self.home_name,
            "team_away": self.away_name,
            "status": self.status,
            "winner": self.winner,
            "type": "soccer_match",
        }


class MatchResponse(BaseModel):
    """Top-level response from the football-data.org API."""

    matches: list[Match] = []


# ---------------------------------------------------------------------------
# ETL state tracker
# ---------------------------------------------------------------------------


class SyncState(BaseModel):
    """Tracks the last successful sync time for incremental ETL."""

    last_sync: datetime | None = None
    match_count: int = 0
