import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # legacy/ -> backend/, so `src.*` resolves

import streamlit as st

from src.core.config import settings
from src.core.errors import RagError
from src.rag.embeddings import get_device, load_embeddings
from src.rag.pipeline import build_combined_chain, ingest_document
from src.rag.rag_chain import answer_query

st.set_page_config(page_title="Document RAG", page_icon="📄", layout="wide")

MAX_ACTIVE_DOCS = 4


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return load_embeddings()


@st.cache_resource(show_spinner=False)
def get_ingested(source_path, file_hash):
    return ingest_document(source_path, get_embeddings())


def save_upload(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()[:16]
    dest = settings.data_dir / uploaded_file.name
    dest.write_bytes(file_bytes)
    return dest, file_hash


st.title("📄 Document RAG")

if "docs" not in st.session_state:
    st.session_state["docs"] = {}  # file_hash -> {"source_path", "entry"}

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            dest_path, file_hash = save_upload(uploaded_file)
            if file_hash not in st.session_state["docs"]:
                try:
                    with st.spinner(f"Processing {uploaded_file.name}…"):
                        entry = get_ingested(str(dest_path), file_hash)
                    st.session_state["docs"][file_hash] = {"source_path": dest_path, "entry": entry}
                except RagError as exc:
                    # Known, expected failure (encrypted/corrupt PDF, blank document) —
                    # show the clear message and skip this file, don't block the others.
                    st.error(f"{uploaded_file.name}: {exc}")
                except Exception:
                    st.error(f"{uploaded_file.name}: something went wrong while processing this document.")

    docs = st.session_state["docs"]
    if docs:
        name_to_hash = {v["entry"]["source_file"]: k for k, v in docs.items()}
        default_selection = st.session_state.get("active_names", list(name_to_hash)[:1])
        default_selection = [n for n in default_selection if n in name_to_hash] or list(name_to_hash)[:1]

        selected_names = st.multiselect(
            f"Active documents (up to {MAX_ACTIVE_DOCS})",
            options=list(name_to_hash),
            default=default_selection,
            max_selections=MAX_ACTIVE_DOCS,
        )

        if selected_names and (
            selected_names != st.session_state.get("active_names") or "pipeline" not in st.session_state
        ):
            st.session_state["active_names"] = selected_names
            entries = [docs[name_to_hash[n]]["entry"] for n in selected_names]
            with st.spinner("Building retriever over selected documents…"):
                try:
                    rag_chain, doc_metadata, *_ = build_combined_chain(entries, get_device())
                    st.session_state["pipeline"] = {"rag_chain": rag_chain, "doc_metadata": doc_metadata}
                    st.session_state.pop("messages", None)
                except RagError as exc:
                    st.session_state.pop("pipeline", None)
                    st.error(str(exc))

    pipeline = st.session_state.get("pipeline")
    if pipeline:
        m = pipeline["doc_metadata"]
        st.divider()
        st.subheader(f"Stats ({', '.join(st.session_state.get('active_names', []))})")
        c1, c2 = st.columns(2)
        c1.metric("Pages", m["total_pages"])
        c2.metric("Tables", m["total_tables"])
        c1.metric("Pictures", m["total_pictures"])
        c2.metric("Text blocks", m["total_text_blocks"])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if not st.session_state.get("pipeline"):
    st.info("Upload one or more PDFs in the sidebar, then pick up to 4 as active documents to start.")
else:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the document…")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        pipeline = st.session_state["pipeline"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = answer_query(question, pipeline["rag_chain"], pipeline["doc_metadata"])
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
