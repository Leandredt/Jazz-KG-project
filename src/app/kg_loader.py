"""Singleton KG loader with Streamlit cache."""
from pathlib import Path
import streamlit as st
from rdflib import Graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPANDED_NT  = PROJECT_ROOT / "kg_artifacts" / "expanded.nt"
INITIAL_TTL  = PROJECT_ROOT / "kg_artifacts" / "initial_kg.ttl"


@st.cache_resource(show_spinner="Loading Knowledge Graph…")
def load_kg() -> Graph:
    g = Graph()
    if EXPANDED_NT.exists():
        g.parse(str(EXPANDED_NT), format="nt")
    elif INITIAL_TTL.exists():
        g.parse(str(INITIAL_TTL), format="turtle")
    from reason.reasoner import Reasoner as _Reasoner
    _r = _Reasoner(g)
    _r.infer_new_facts()
    return g
