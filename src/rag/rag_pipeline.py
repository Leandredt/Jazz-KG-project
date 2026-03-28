"""
Retrieval-Augmented Generation (RAG) Pipeline
==============================================
Retrieves relevant facts from an RDF Knowledge Graph using SPARQL and keyword
matching, then formats a structured natural-language response — no external LLM
required.

Interface:
    RAGPipeline(kg: Graph, llm_model_name: str = "kg-rag")
    .retrieve_info(query: str) -> str
    .generate_response(prompt: str, retrieved_context: str) -> str
    .run_rag_pipeline(query: str, prompt: str) -> str
"""

import logging
import re
from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal
from rdflib.plugins.sparql import prepareQuery

logger = logging.getLogger(__name__)

JAZZ = Namespace("http://jazz-kg.org/ontology#")
WD = Namespace("http://www.wikidata.org/entity/")
JAZZ_RESOURCE = "http://jazz-kg.org/resource/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_of(g: Graph, uri: URIRef) -> str:
    """Return the rdfs:label of a URI, or a slug derived from the URI."""
    for obj in g.objects(uri, RDFS.label):
        return str(obj)
    # Derive from URI
    return str(uri).rstrip("/").split("/")[-1].replace("_", " ")


def _type_label(g: Graph, uri: URIRef) -> str:
    """Return a human-readable type name for a resource."""
    type_map = {
        str(JAZZ.Musician): "Musician",
        str(JAZZ.Album): "Album",
        str(JAZZ.RecordLabel): "Record Label",
        str(JAZZ.Band): "Band",
        str(JAZZ.Location): "Location",
        str(JAZZ.Instrument): "Instrument",
        str(JAZZ.Genre): "Genre",
    }
    for _, _, cls in g.triples((uri, RDF.type, None)):
        t = type_map.get(str(cls))
        if t:
            return t
    return "Entity"


def _keywords_from_query(query: str) -> list[str]:
    """Extract meaningful keywords from a natural-language query."""
    stop_words = {
        "who", "what", "where", "when", "how", "is", "was", "were", "are",
        "did", "do", "does", "a", "an", "the", "of", "in", "on", "at", "to",
        "for", "and", "or", "tell", "me", "about", "describe", "list", "show",
        "which", "that", "this", "their", "his", "her", "its", "with",
    }
    tokens = re.findall(r"[A-Za-z']+", query.lower())
    keywords = [t for t in tokens if t not in stop_words and len(t) > 2]
    return keywords


def _find_matching_entities(g: Graph, keywords: list[str]) -> list[URIRef]:
    """Find entities whose labels contain any of the query keywords."""
    matches: list[tuple[int, URIRef]] = []
    for subj, _, label_lit in g.triples((None, RDFS.label, None)):
        if not isinstance(subj, URIRef):
            continue
        if JAZZ_RESOURCE not in str(subj):
            continue
        label_lower = str(label_lit).lower()
        score = sum(1 for kw in keywords if kw in label_lower)
        if score > 0:
            matches.append((score, subj))
    # Return top-10 by score
    matches.sort(key=lambda x: -x[0])
    return [uri for _, uri in matches[:10]]


def _describe_entity(g: Graph, uri: URIRef) -> list[str]:
    """Return a list of human-readable fact sentences for a KG entity."""
    facts: list[str] = []
    label = _label_of(g, uri)
    etype = _type_label(g, uri)

    prop_templates = {
        str(JAZZ.bornIn): "{subject} was born in {object}.",
        str(JAZZ.plays): "{subject} plays the {object}.",
        str(JAZZ.memberOf): "{subject} is/was a member of {object}.",
        str(JAZZ.signedWith): "{subject} was signed with {object}.",
        str(JAZZ.recordedOn): "{subject} recorded the album {object}.",
        str(JAZZ.playedBy): "{subject} was performed by {object}.",
        str(JAZZ.releasedBy): "{subject} was released by {object}.",
        str(JAZZ.hasGenre): "{subject} plays/is associated with the genre {object}.",
        str(JAZZ.basedIn): "{subject} is based in {object}.",
        str(JAZZ.influencedBy): "{subject} was influenced by {object}.",
        str(RDFS.label): None,  # skip raw label triples
    }

    for _, pred, obj in g.triples((uri, None, None)):
        if pred == RDF.type or pred == RDFS.label:
            continue
        template = prop_templates.get(str(pred))
        if template is None:
            continue  # skip unknown/raw predicates
        if isinstance(obj, URIRef):
            obj_label = _label_of(g, obj)
        else:
            obj_label = str(obj)
        facts.append(template.format(subject=label, object=obj_label))

    return facts


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    KG-grounded RAG pipeline.

    Parameters
    ----------
    kg : rdflib.Graph
        The knowledge graph to retrieve from.
    llm_model_name : str
        Kept for interface compatibility; no external LLM is called.
    """

    def __init__(self, kg: Graph, llm_model_name: str = "kg-rag"):
        self.kg = kg
        self.llm_model_name = llm_model_name
        logger.info("RAGPipeline initialised — %d triples in KG.", len(kg))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_info(self, query: str) -> str:
        """
        Search the KG for entities matching query keywords.
        Returns a multi-sentence context string with relevant facts.
        """
        logger.info("Retrieving for query: %r", query)
        keywords = _keywords_from_query(query)
        if not keywords:
            return "No query keywords could be extracted."

        matched_entities = _find_matching_entities(self.kg, keywords)
        if not matched_entities:
            return f"No entities found in the KG matching keywords: {keywords}."

        context_parts: list[str] = []
        for uri in matched_entities:
            label = _label_of(self.kg, uri)
            etype = _type_label(self.kg, uri)
            facts = _describe_entity(self.kg, uri)
            header = f"[{etype}] {label}"
            if facts:
                context_parts.append(header + ": " + " ".join(facts[:5]))
            else:
                context_parts.append(header + " (no further facts available).")

        return "\n".join(context_parts)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_response(self, prompt: str, retrieved_context: str) -> str:
        """
        Format a structured response from the retrieved KG context.
        No external LLM is used; the context itself is the answer.
        """
        logger.info("Generating response from retrieved context.")

        if not retrieved_context or retrieved_context.startswith("No "):
            return (
                f"Prompt: {prompt}\n\n"
                "No relevant information was found in the Jazz Knowledge Graph for this query."
            )

        response_lines = [
            f"Query: {prompt}",
            "",
            "Knowledge Graph findings:",
            "─" * 50,
            retrieved_context,
            "─" * 50,
            "Summary: The above facts were retrieved directly from the Jazz KG.",
        ]
        return "\n".join(response_lines)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_rag_pipeline(self, query: str, prompt: str) -> str:
        """Execute retrieval then generation and return the final response."""
        retrieved_context = self.retrieve_info(query)
        response = self.generate_response(prompt, retrieved_context)
        return response


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    g = Graph()
    kg_path = PROJECT_ROOT / "kg_artifacts" / "initial_kg.ttl"
    if kg_path.exists():
        g.parse(str(kg_path), format="turtle")
        logger.info("Loaded KG: %d triples", len(g))
    else:
        logger.warning("No KG file found at %s; using empty graph.", kg_path)

    rag = RAGPipeline(g)

    examples = [
        ("Who is Miles Davis and what instrument did he play?",
         "Provide a concise biography of this artist."),
        ("Tell me about Blue Note Records",
         "Describe this record label."),
        ("What albums did John Coltrane record?",
         "List the albums associated with this musician."),
    ]

    for query, prompt in examples:
        print("\n" + "=" * 60)
        print(f"QUERY : {query}")
        result = rag.run_rag_pipeline(query, prompt)
        print(result)
