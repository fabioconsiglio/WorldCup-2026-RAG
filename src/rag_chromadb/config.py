"""Centralized configuration for the RAG ChromaDB project.

All settings are loaded from environment variables with sensible defaults
for a local LM Studio + ChromaDB setup.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API / LLM settings (LM Studio local server)
# ---------------------------------------------------------------------------
OPENAI_BASE_URL: str = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
OPENAI_API_KEY: str = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "qwen3.5-2b")

# ---------------------------------------------------------------------------
# ChromaDB settings
# ---------------------------------------------------------------------------
# Resolve relative paths against the project root so the DB location is stable
# regardless of the CWD the app is launched from. (Root cause of empty-query
# bugs: ./local_db resolved to a different dir per script, splitting data
# across two databases.)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BASE_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    str(_PROJECT_ROOT / "local_db"),
)
CHROMA_DB_PATH: str = str(Path(_BASE_DB_PATH).expanduser())
if not Path(CHROMA_DB_PATH).is_absolute():
    CHROMA_DB_PATH = str((_PROJECT_ROOT / CHROMA_DB_PATH).resolve())

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
