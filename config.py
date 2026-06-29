"""Centralized configuration for the RAG ChromaDB project.

All settings are loaded from environment variables with sensible defaults
for a local LM Studio + ChromaDB setup.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# API / LLM settings (LM Studio local server)
# ---------------------------------------------------------------------------
OPENAI_BASE_URL: str = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
OPENAI_API_KEY: str = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "mistral-7b-instruct")

# ---------------------------------------------------------------------------
# ChromaDB settings
# ---------------------------------------------------------------------------
CHROMA_DB_PATH: str = os.getenv(
    "CHROMA_DB_PATH",
    str(Path(__file__).resolve().parent / "local_db"),
)

# ---------------------------------------------------------------------------
# Collection names
# ---------------------------------------------------------------------------
DEFAULT_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "my_documents")
WC_COLLECTION: str = os.getenv("WC_COLLECTION", "world_cup_2026")

# ---------------------------------------------------------------------------
# External API tokens
# ---------------------------------------------------------------------------
FOOTBALL_DATA_API_TOKEN: str = os.getenv("FOOTBALL_DATA_API_TOKEN", "")
FOOTBALL_DATA_API_URL: str = os.getenv(
    "FOOTBALL_DATA_API_URL",
    "https://api.football-data.org/v4/competitions/WC/matches",
)

# ---------------------------------------------------------------------------
# ETL state
# ---------------------------------------------------------------------------
ETL_STATE_FILE: str = os.getenv(
    "ETL_STATE_FILE",
    str(Path(__file__).resolve().parent / "etl_state.json"),
)

# ---------------------------------------------------------------------------
# RAG defaults
# ---------------------------------------------------------------------------
DEFAULT_N_RESULTS: int = int(os.getenv("RAG_N_RESULTS", "3"))
DEFAULT_TEMPERATURE: float = float(os.getenv("RAG_TEMPERATURE", "0.3"))