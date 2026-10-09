"""
Interactive visualization layer (PRD 2.1: "interactive visualization scripts
using Streamlit and PyVis to render real-time node connections during graph
retrieval").

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import tempfile
import pypdf
from pyvis.network import Network
import streamlit as st

from app.core.config import get_settings
from app.core.graph_store import GraphStore
from app.core.ingestion import chunk_document, extract_triples_for_document
from app.core.llm_provider import get_chat_model
from app.core.retrieval import HybridRetriever, naive_entity_extraction
from app.core.self_correction_graph import run_self_correcting_query
from app.core.vector_store import VectorStore

st.set_page_config(page_title="Graph-Augmented Self-RAG Engine", layout="wide")

settings = get_settings()


@st.cache_resource
def get_stores():
    vs = VectorStore(settings)
    gs = GraphStore(persist_path=settings.graph_persist_path)
    return vs, gs


vector_store, graph_store = get_stores()
retriever = HybridRetriever(vector_store, graph_store, settings)

st.title("🕸️ Graph-Augmented Self-RAG Engine")
st.caption(
    f"LLM provider: `{settings.llm_provider}` (`{settings.groq_chat_model}`) · Graph nodes: {graph_store.stats()['nodes']} · "
    f"Graph edges: {graph_store.stats()['edges']} · Vector chunks: {vector_store.count()}"
)

tab_ingest, tab_query, tab_graph = st.tabs(["📥 Ingest Document / PDF", "🔎 Query", "🌐 Full Graph"])

with tab_ingest:
    st.subheader("Ingest a Document or PDF File")
    
    col_a, col_b = st.columns(2)
    with col_a:
        doc_id = st.text_input("Document ID", value="pdf_document_1")
    with col_b:
        uploaded_file = st.file_uploader("Upload File (PDF, TXT, MD)", type=["pdf", "txt", "md"])
    
    extracted_text = ""
    source_name = "pasted_text.txt"

    if uploaded_file is not None:
        source_name = uploaded_file.name
        if uploaded_file.name.endswith(".pdf"):
            try:
                reader = pypdf.PdfReader(uploaded_file)
                pdf_pages = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        pdf_pages.append(t)
                extracted_text = "\n\n".join(pdf_pages)
                st.info(f"Extracted {len(reader.pages)} pages from `{uploaded_file.name}` ({len(extracted_text)} characters).")
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")
        else:
            extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.info(f"Loaded file `{uploaded_file.name}` ({len(extracted_text)} characters).")

    text = st.text_area("Document Content (extracted or pasted)", value=extracted_text, height=250)

    if st.button("Ingest & Build Knowledge Graph", type="primary"):
        if not text.strip():
            st.error("Please provide non-empty document text or upload a valid file.")
        else:
            with st.spinner("Chunking text, extracting triples via LLM, and populating Knowledge Graph & Vector Store..."):
                llm = get_chat_model(settings)
                chunks = chunk_document(doc_id, source_name, text, settings)
                chunks = extract_triples_for_document(chunks, llm)
                vector_store.add_chunks(chunks)
                graph_store.ingest_chunks(chunks)
                graph_store.save()
            st.success(f"Successfully ingested `{source_name}`! Created {len(chunks)} chunks and {sum(len(c.triples) for c in chunks)} Knowledge Triples.")
            st.rerun()

with tab_query:
    st.subheader("Ask a Multi-Hop Question")
    q = st.text_input("Query", value="How does Component A indirectly affect Component C?")
    if st.button("Run Self-RAG Pipeline", type="primary"):
        with st.spinner("Retrieving via Hybrid CRI, generating answer, and running self-correction guardrails..."):
            llm = get_chat_model(settings)
            final_state = run_self_correcting_query(q, retriever, llm, settings)

        st.markdown("### 📝 Answer")
        st.write(final_state["answer"])

        ev = final_state["evaluation"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Hallucination score", f"{ev.hallucination_score:.2f}" if ev else "n/a")
        col2.metric("Relevance score", f"{ev.relevance_score:.2f}" if ev else "n/a")
        col3.metric("Retries used", final_state["retries_used"])

        st.markdown("### 📚 Retrieved Contexts (Ranked by Composite Relevance Index)")
        for c in final_state["contexts"]:
            with st.expander(f"[{c.chunk_id}] CRI={c.composite_relevance_index:.4f} · hops={c.graph_hops_used}"):
                st.write(c.text)
                st.caption(f"vector_score={c.vector_score} · pagerank={c.pagerank_score} · source={c.source}")

        st.markdown("### ⚙️ Execution Trace Log")
        for step in final_state["trace"]:
            st.text(f"[{step.node}] {step.detail}")

        # --- Graph visualization of seed nodes + 2-hop neighborhood -------
        seed_nodes = naive_entity_extraction(final_state["current_query"], graph_store)
        hop_map = graph_store.bfs_k_hop(seed_nodes, hops=settings.graph_hops)
        if hop_map:
            st.markdown("### 🕸️ Retrieval Subgraph (Seed Nodes + 2-Hop Neighborhood)")
            net = Network(height="500px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
            pr_scores = graph_store.pagerank(damping=settings.pagerank_damping)
            for node, hop in hop_map.items():
                color = "#e74c3c" if hop == 0 else ("#3498db" if hop == 1 else "#2ecc71")
                size = 15 + 100 * pr_scores.get(node, 0)
                net.add_node(node, label=node, color=color, size=size, title=f"PageRank={pr_scores.get(node, 0):.4f}")
            for u, v, data in graph_store.graph.edges(data=True):
                if u in hop_map and v in hop_map:
                    net.add_edge(u, v, title=", ".join(data.get("predicates", [])))
            output_file = os.path.join(tempfile.gettempdir(), "graph_viz.html")
            net.save_graph(output_file)
            with open(output_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=520)
        else:
            st.info("No seed graph entities matched this query directly -- answer relied on dense vector matches.")

with tab_graph:
    st.subheader("🌐 Full Knowledge Graph Visualization")
    if graph_store.graph.number_of_nodes() == 0:
        st.info("Knowledge Graph is currently empty. Ingest a document or PDF first!")
    else:
        pr_scores = graph_store.pagerank(damping=settings.pagerank_damping)
        net = Network(height="600px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
        for node in graph_store.graph.nodes:
            size = 15 + 150 * pr_scores.get(node, 0)
            net.add_node(node, label=node, size=size, title=f"PageRank={pr_scores.get(node, 0):.4f}")
        for u, v, data in graph_store.graph.edges(data=True):
            net.add_edge(u, v, title=", ".join(data.get("predicates", [])))
        output_file = os.path.join(tempfile.gettempdir(), "full_graph_viz.html")
        net.save_graph(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=620)
