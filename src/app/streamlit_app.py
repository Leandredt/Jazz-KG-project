"""Jazz KG — Streamlit demo interface (3-tab layout).

Launch with:
    streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from rdflib import Graph, Namespace
from rdflib.namespace import OWL, RDF, RDFS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.kg_loader import load_kg  # noqa: E402

JAZZ = Namespace("http://jazz-kg.org/ontology#")
WDT  = Namespace("http://www.wikidata.org/prop/direct/")

PRED_LABELS = {
    "type": "Type", "label": "Name", "sameAs": "Wikidata link",
    "plays": "Plays instrument", "bornIn": "Born in", "memberOf": "Member of",
    "releasedBy": "Released by", "hasGenre": "Genre", "influencedBy": "Influenced by",
    "signedWith": "Signed with", "recordedOn": "Recorded on", "basedIn": "Based in",
    "P569": "Birth date", "P19": "Place of birth", "P175": "Performer",
    "P264": "Record label", "P136": "Genre", "P1303": "Instrument",
    "P463": "Member of", "P737": "Influenced by", "P31": "Instance of",
}

st.set_page_config(page_title="Jazz Knowledge Graph", page_icon="🎵", layout="wide")

# ---------------------------------------------------------------------------
# RAG import
# ---------------------------------------------------------------------------
try:
    from rag.nl_sparql import NLToSPARQL, SUGGESTED_QUESTIONS
    _rag_available = True
except ImportError:
    _rag_available = False
    SUGGESTED_QUESTIONS = [
        "Who are the musicians that play trumpet?",
        "What albums did Miles Davis record?",
        "Which musicians were born in New Orleans?",
        "Who influenced John Coltrane?",
        "What genres did Charlie Parker play?",
    ]

# ---------------------------------------------------------------------------
# Load KG
# ---------------------------------------------------------------------------
g: Graph = load_kg()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _subject_counts() -> list[tuple[str, int]]:
    counts = Counter(str(s) for s in g.subjects() if "jazz-kg.org/resource" in str(s))
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)

@st.cache_data(show_spinner=False)
def _alignment_count() -> int:
    path = PROJECT_ROOT / "kg_artifacts" / "alignment.ttl"
    if not path.exists():
        return -1
    ag = Graph()
    ag.parse(str(path), format="turtle")
    return sum(1 for _ in ag.triples((None, OWL.sameAs, None)))

@st.cache_data(show_spinner=False)
def _kge_metrics() -> dict | None:
    p = PROJECT_ROOT / "reports" / "kge_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None

def _fmt_file(path: Path) -> str:
    return f"{path.stat().st_size // 1024} KB" if path.exists() else "missing"

def _friendly_pred(raw: str) -> str:
    key = raw.split("/")[-1].split("#")[-1]
    return PRED_LABELS.get(key, key)

def _friendly_obj(raw: str) -> str:
    if raw.startswith("http"):
        return raw.split("/")[-1].split("#")[-1].replace("_", " ")
    return raw[:100]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_rag, tab_stats, tab_graph = st.tabs(["🎵 RAG Demo", "📊 Overview & Stats", "🗺 Graph Explorer"])

# ===========================================================================
# TAB 1 — RAG Demo
# ===========================================================================
with tab_rag:
    st.title("Ask the Jazz Knowledge Graph")
    st.caption("Ask a question in plain English — the system generates a SPARQL query and searches the KG.")

    st.markdown("**Suggested questions — click to use:**")
    cols = st.columns(min(len(SUGGESTED_QUESTIONS), 4))
    for i, q in enumerate(SUGGESTED_QUESTIONS[:4]):
        if cols[i].button(q, key=f"sq_{i}"):
            st.session_state["rag_input"] = q
    if len(SUGGESTED_QUESTIONS) > 4:
        cols2 = st.columns(min(len(SUGGESTED_QUESTIONS) - 4, 4))
        for i, q in enumerate(SUGGESTED_QUESTIONS[4:8]):
            if cols2[i].button(q, key=f"sq2_{i}"):
                st.session_state["rag_input"] = q

    question = st.text_area(
        "Your question",
        height=80,
        placeholder="e.g. What albums did Miles Davis record?",
        key="rag_input",
    )

    if st.button("Ask", type="primary", key="ask_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not _rag_available:
            st.error("RAG pipeline not available.")
        else:
            with st.spinner("Generating SPARQL query and searching the knowledge graph…"):
                try:
                    pipeline = NLToSPARQL(g)
                    result = pipeline.answer(question)

                    sparql    = result.get("sparql", "")
                    answer    = result.get("answer_text", "")
                    n_repairs = result.get("repairs_needed", 0)
                    success   = result.get("success", False)
                    n_results = len(result.get("results", []))

                    if sparql:
                        with st.expander("Generated SPARQL query", expanded=True):
                            st.code(sparql, language="sparql")

                    st.subheader("Answer")
                    st.write(answer if answer else "_No results found._")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Results found", n_results)
                    col2.metric("Query executed", "✅" if success else "❌")
                    col3.metric("Self-repairs", n_repairs)

                    if n_repairs > 0:
                        st.info("The SPARQL query was automatically corrected by the self-repair mechanism.")

                except Exception as e:
                    st.error(f"Error: {e}")

    eval_path = PROJECT_ROOT / "reports" / "rag_evaluation.json"
    if eval_path.exists():
        with st.expander("📋 Baseline vs RAG Evaluation (7 questions)"):
            data = json.loads(eval_path.read_text())
            if isinstance(data, list):
                st.dataframe(pd.DataFrame(data), hide_index=True)
            else:
                st.json(data)

# ===========================================================================
# TAB 2 — Overview & Stats
# ===========================================================================
with tab_stats:
    st.title("Knowledge Graph — Overview & Statistics")

    aln   = _alignment_count()
    kge   = _kge_metrics()
    top20 = _subject_counts()[:20]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total triples",       f"{len(g):,}")
    c2.metric("Unique entities",     f"{len(_subject_counts()):,}")
    c3.metric("Wikidata alignments", f"{aln:,}" if aln >= 0 else "N/A")
    c4.metric("KGE models",          "TransE + DistMult" if kge else "0")

    if kge:
        st.subheader("KGE Metrics")
        rows = []
        for model, metrics in kge.items():
            if isinstance(metrics, dict):
                rows.append({"Model": model, **metrics})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader("Generated files")
    rows = []
    for name in ["crawler_output.jsonl", "extracted_knowledge.csv"]:
        p = PROJECT_ROOT / "data" / name
        rows.append({"File": f"data/{name}", "Status": _fmt_file(p)})
    for name in ["ontology.ttl", "initial_kg.ttl", "alignment.ttl", "expanded.nt"]:
        p = PROJECT_ROOT / "kg_artifacts" / name
        rows.append({"File": f"kg_artifacts/{name}", "Status": _fmt_file(p)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    col_l, col_r = st.columns(2)

    with col_l:
        if top20:
            st.subheader("Top 20 Entities")
            labels = [u.split("/")[-1].replace("_", " ")[:20] for u, _ in top20]
            counts = [c for _, c in top20]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.barh(labels[::-1], counts[::-1], color="steelblue")
            ax.set_xlabel("Triples")
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)

    with col_r:
        csv_path = PROJECT_ROOT / "data" / "extracted_knowledge.csv"
        if csv_path.exists():
            st.subheader("Entity Type Distribution")
            df_csv = pd.read_csv(csv_path)
            if "entity_type" in df_csv.columns:
                td = df_csv["entity_type"].value_counts()
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                ax2.pie(td.values, labels=td.index, autopct="%1.1f%%", startangle=140)
                plt.tight_layout()
                st.pyplot(fig2)

    birth_years = []
    for _, _, o in g.triples((None, WDT.P569, None)):
        m = re.search(r"(\d{4})", str(o))
        if m:
            y = int(m.group(1))
            if 1850 <= y <= 2010:
                birth_years.append(y)
    if birth_years:
        st.subheader("Musician Birth Years")
        col_hist, _ = st.columns([2, 1])
        with col_hist:
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            ax3.hist(birth_years, bins=30, color="coral", edgecolor="white")
            ax3.set_xlabel("Year")
            ax3.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig3)

    tsne_path = PROJECT_ROOT / "reports" / "tsne_embeddings.png"
    if tsne_path.exists():
        st.subheader("KGE Embeddings (PCA projection)")
        col_t, _ = st.columns([2, 1])
        with col_t:
            st.image(str(tsne_path))

# ===========================================================================
# TAB 3 — Graph Explorer
# ===========================================================================
with tab_graph:
    st.title("Graph Explorer")
    st.caption("Explore connections between artists, albums, labels and genres.")

    top_entities = [u.split("/")[-1].replace("_", " ") for u, _ in _subject_counts()[:15]]

    col_a, col_b = st.columns([2, 1])
    with col_a:
        entity_input = st.text_input(
            "Search an artist, album, or label",
            value="Miles Davis",
            help="Type any name — musician, album, record label, city, genre…",
        )
    with col_b:
        pick = st.selectbox("Or pick from top entities", ["—"] + top_entities)

    if pick != "—":
        entity_input = pick

    max_edges = st.slider("Number of connections to show", 5, 50, 15)

    target = None
    search = entity_input.lower().replace(" ", "_")
    for s in g.subjects():
        if search in str(s).lower():
            target = s
            break

    if target:
        center = str(target).split("/")[-1].replace("_", " ")
        st.info(f"Showing: **{center}**")

        try:
            from pyvis.network import Network
            import streamlit.components.v1 as components

            net = Network(height="480px", width="100%", bgcolor="#1a1a2e",
                          font_color="white", directed=True, notebook=False)
            net.set_options("""{
              "physics": {"barnesHut": {"gravitationalConstant": -5000,
                                        "springLength": 150,
                                        "springConstant": 0.02}},
              "nodes": {"font": {"size": 14}},
              "interaction": {"zoomSpeed": 0.3, "navigationButtons": true}
            }""")
            net.add_node(center, label=center, color="#4FC3F7", size=30)
            for _, p, o in list(g.triples((target, None, None)))[:max_edges]:
                pred = _friendly_pred(str(p))
                obj  = _friendly_obj(str(o))
                if obj and obj != center:
                    net.add_node(obj, label=obj, color="#FFB74D", size=18)
                    net.add_edge(center, obj, title=pred, label=pred,
                                 font={"size": 10, "align": "middle"})

            html_path = PROJECT_ROOT / "reports" / "kg_graph.html"
            html_path.parent.mkdir(exist_ok=True)
            net.save_graph(str(html_path))
            with open(html_path) as f:
                components.html(f.read(), height=500, scrolling=False)

        except ImportError:
            import networkx as nx
            G = nx.DiGraph()
            G.add_node(center)
            for _, p, o in list(g.triples((target, None, None)))[:max_edges]:
                G.add_edge(center, _friendly_obj(str(o)))
            fig, ax = plt.subplots(figsize=(9, 6))
            pos = nx.spring_layout(G, seed=42, k=2)
            nx.draw(G, pos, with_labels=True, node_color="lightblue",
                    node_size=1500, font_size=9, ax=ax, arrows=True)
            st.pyplot(fig)

        st.subheader(f"Properties of {center}")
        props = [{"Property": _friendly_pred(str(p)), "Value": _friendly_obj(str(o))}
                 for _, p, o in g.triples((target, None, None)) if _friendly_obj(str(o))]
        if props:
            st.dataframe(pd.DataFrame(props), hide_index=True)
    else:
        st.warning("No entity found. Try a different name.")
