"""Streamlit frontend for the local RAG chatbot."""

from __future__ import annotations

import streamlit as st

from rag_chromadb.config import DEFAULT_N_RESULTS, WC_COLLECTION
from rag_chromadb.db import (
    build_rag_prompt,
    generate_answer,
    get_collection,
    query_collection,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Local RAG Chat", page_icon="🧠")
st.title("Local RAG Chat 🧠")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input & RAG pipeline
# ---------------------------------------------------------------------------
if user_query := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve
    collection = get_collection(WC_COLLECTION)
    context_chunks = query_collection(
        collection, user_query, n_results=DEFAULT_N_RESULTS
    )

    # Generate
    prompt = build_rag_prompt(user_query, context_chunks)

    with st.chat_message("assistant"):
        stream = generate_answer(prompt, stream=True)
        if stream:
            response = st.write_stream(stream)
        else:
            response = "Sorry, something went wrong generating the answer."

    st.session_state.messages.append({"role": "assistant", "content": response})
