"""Tests for db.py helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag_chromadb.db import build_rag_prompt, query_collection


class TestBuildRagPrompt:
    def test_single_chunk(self) -> None:
        prompt = build_rag_prompt("What is X?", ["X is 1."])
        assert "What is X?" in prompt
        assert "X is 1." in prompt
        assert "Context:" in prompt
        assert "Query:" in prompt

    def test_multiple_chunks_joined(self) -> None:
        prompt = build_rag_prompt("Q", ["A.", "B."])
        assert "A.\n\nB." in prompt


class TestQueryCollection:
    @patch("rag_chromadb.db.get_embedding")
    def test_returns_document_texts(self, mock_embed: MagicMock) -> None:
        mock_embed.return_value = [0.1, 0.2]
        col = MagicMock()
        col.query.return_value = {"documents": [["chunk A", "chunk B"]]}

        result = query_collection(col, "any query", n_results=2)
        assert result == ["chunk A", "chunk B"]

    @patch("rag_chromadb.db.get_embedding")
    def test_empty_results_returns_empty_list(self, mock_embed: MagicMock) -> None:
        mock_embed.return_value = [0.1, 0.2]
        col = MagicMock()
        col.query.return_value = {"documents": [[]]}

        result = query_collection(col, "any query")
        assert result == []

    @patch("rag_chromadb.db.get_embedding")
    def test_no_documents_key_returns_empty_list(self, mock_embed: MagicMock) -> None:
        mock_embed.return_value = [0.1, 0.2]
        col = MagicMock()
        col.query.return_value = {}

        result = query_collection(col, "any query")
        assert result == []
