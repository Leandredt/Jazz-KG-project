"""
Schema Summary
==============
Builds a compact text description of the Jazz KG schema for injection into
LLM prompts. Includes OWL class counts, predicates with examples, and
sample entity labels per class.
"""

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef

JAZZ = Namespace("http://jazz-kg.org/ontology#")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
WDE = Namespace("http://www.wikidata.org/entity/")
JAZZR = Namespace("http://jazz-kg.org/resource/")

# Known OWL classes in the ontology
CLASSES = {
    "jazz:Musician":     str(JAZZ.Musician),
    "jazz:Album":        str(JAZZ.Album),
    "jazz:RecordLabel":  str(JAZZ.RecordLabel),
    "jazz:Band":         str(JAZZ.Band),
    "jazz:Location":     str(JAZZ.Location),
    "jazz:Instrument":   str(JAZZ.Instrument),
    "jazz:Genre":        str(JAZZ.Genre),
}

# Key ontology predicates with human-readable labels
ONTOLOGY_PREDICATES = {
    str(JAZZ.bornIn):      "jazz:bornIn",
    str(JAZZ.plays):       "jazz:plays",
    str(JAZZ.memberOf):    "jazz:memberOf",
    str(JAZZ.signedWith):  "jazz:signedWith",
    str(JAZZ.recordedOn):  "jazz:recordedOn",
    str(JAZZ.playedBy):    "jazz:playedBy",
    str(JAZZ.releasedBy):  "jazz:releasedBy",
    str(JAZZ.hasGenre):    "jazz:hasGenre",
    str(JAZZ.basedIn):     "jazz:basedIn",
    str(JAZZ.influencedBy): "jazz:influencedBy",
}

# Wikidata predicates present in the expanded KG
WIKIDATA_PREDICATES = {
    str(WDT.P31):   "wdt:P31 (instance of)",
    str(WDT.P175):  "wdt:P175 (performer)",
    str(WDT.P264):  "wdt:P264 (record label)",
    str(WDT.P1303): "wdt:P1303 (instrument)",
    str(WDT.P136):  "wdt:P136 (genre)",
    str(WDT.P19):   "wdt:P19 (place of birth)",
    str(WDT.P569):  "wdt:P569 (date of birth)",
    str(WDT.P737):  "wdt:P737 (influenced by)",
    str(WDT.P495):  "wdt:P495 (country of origin)",
}


def _label_of(g: Graph, uri: URIRef) -> str:
    """Return rdfs:label or a URI-derived slug."""
    for lbl in g.objects(uri, RDFS.label):
        s = str(lbl)
        # Prefer English labels
        if hasattr(lbl, "language") and lbl.language == "en":
            return s
        return s
    return str(uri).rstrip("/").split("/")[-1].replace("_", " ")


def build_schema_summary(g: Graph) -> str:
    """
    Return a compact multi-section text describing the KG schema.

    Sections:
      1. Class inventory with triple counts
      2. Ontology predicates with example values
      3. Wikidata predicates found in expanded KG
      4. Sample entities per class (up to 5)
    """
    lines: list[str] = []

    # ------------------------------------------------------------------
    # 1. Class counts
    # ------------------------------------------------------------------
    lines.append("=== KG Classes (jazz-kg ontology) ===")
    for prefix, class_uri in CLASSES.items():
        count = sum(1 for _ in g.triples((None, RDF.type, URIRef(class_uri))))
        lines.append(f"  {prefix}  — {count} instances")
    lines.append("")

    # ------------------------------------------------------------------
    # 2. Ontology predicates with one example value each
    # ------------------------------------------------------------------
    lines.append("=== Ontology Predicates (examples) ===")
    for pred_uri, pred_label in ONTOLOGY_PREDICATES.items():
        example_parts = []
        for s, _, o in g.triples((None, URIRef(pred_uri), None)):
            s_lbl = _label_of(g, s) if isinstance(s, URIRef) else str(s)
            o_lbl = _label_of(g, o) if isinstance(o, URIRef) else str(o)
            example_parts.append(f'"{s_lbl}" → "{o_lbl}"')
            if len(example_parts) >= 2:
                break
        if example_parts:
            lines.append(f"  {pred_label}: {'; '.join(example_parts)}")
        else:
            lines.append(f"  {pred_label}: (no data)")
    lines.append("")

    # ------------------------------------------------------------------
    # 3. Wikidata predicates
    # ------------------------------------------------------------------
    lines.append("=== Wikidata Predicates in Expanded KG ===")
    for pred_uri, pred_label in WIKIDATA_PREDICATES.items():
        count = sum(1 for _ in g.triples((None, URIRef(pred_uri), None)))
        example = ""
        for s, _, o in g.triples((None, URIRef(pred_uri), None)):
            s_lbl = _label_of(g, s) if isinstance(s, URIRef) else str(s)
            o_lbl = _label_of(g, o) if isinstance(o, URIRef) else str(o)
            example = f'e.g. "{s_lbl}" → "{o_lbl}"'
            break
        lines.append(f"  {pred_label}  ({count} triples)  {example}")
    lines.append("")

    # ------------------------------------------------------------------
    # 4. Sample entities per class
    # ------------------------------------------------------------------
    lines.append("=== Sample Entities per Class ===")
    for prefix, class_uri in CLASSES.items():
        samples = []
        for subj, _, _ in g.triples((None, RDF.type, URIRef(class_uri))):
            if not isinstance(subj, URIRef):
                continue
            lbl = _label_of(g, subj)
            samples.append(lbl)
            if len(samples) >= 5:
                break
        if samples:
            lines.append(f"  {prefix}: {', '.join(samples)}")
        else:
            lines.append(f"  {prefix}: (none found)")
    lines.append("")

    # ------------------------------------------------------------------
    # 5. Namespace guide
    # ------------------------------------------------------------------
    lines.append("=== Namespace Prefixes ===")
    lines.append("  PREFIX jazz:  <http://jazz-kg.org/ontology#>")
    lines.append("  PREFIX jazzr: <http://jazz-kg.org/resource/>")
    lines.append("  PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>")
    lines.append("  PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>")
    lines.append("  PREFIX wdt:   <http://www.wikidata.org/prop/direct/>")
    lines.append("  PREFIX wd:    <http://www.wikidata.org/entity/>")
    lines.append("  PREFIX owl:   <http://www.w3.org/2002/07/owl#>")

    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    g = Graph()
    for fname in ("expanded.nt", "initial_kg.ttl", "ontology.ttl"):
        p = PROJECT_ROOT / "kg_artifacts" / fname
        if p.exists():
            fmt = "nt" if fname.endswith(".nt") else "turtle"
            g.parse(str(p), format=fmt)
    print(build_schema_summary(g))
