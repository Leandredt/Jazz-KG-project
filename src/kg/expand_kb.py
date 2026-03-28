"""
Lab 2 - KB Expansion via SPARQL
=================================
Expands the aligned KG by querying Wikidata SPARQL for broad jazz-domain
triples, targeting ≥50k triples total.

Strategy:
  1. Get all jazz musicians from Wikidata (genre=Q8341) with their properties
  2. Get all jazz albums with their performer/label/date
  3. Get all jazz record labels and their properties
  4. 1-hop expansion on aligned QIDs
  5. Merge with initial KG + alignment

Usage:
    python src/kg/expand_kb.py

Output:
    kg_artifacts/expanded.nt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import time

import requests
from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal, URIRef, BNode

from ontology import JAZZ, build_ontology

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INITIAL_KG_PATH = PROJECT_ROOT / "kg_artifacts" / "initial_kg.ttl"
ALIGNMENT_PATH  = PROJECT_ROOT / "kg_artifacts" / "alignment.ttl"
OUTPUT_PATH     = PROJECT_ROOT / "kg_artifacts" / "expanded.nt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expand_kb")

WD       = Namespace("http://www.wikidata.org/entity/")
WDT      = Namespace("http://www.wikidata.org/prop/direct/")
WIKIBASE = Namespace("http://wikiba.se/ontology#")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
REQUEST_DELAY   = 2.0   # polite delay between SPARQL calls


# ---------------------------------------------------------------------------
# SPARQL helper with retry
# ---------------------------------------------------------------------------

def sparql_query(query: str, session: requests.Session, timeout: int = 45) -> list[dict]:
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "JazzKGExpander/1.0 (academic; ESILV course)"
    }
    for attempt in range(3):
        try:
            resp = session.get(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("results", {}).get("bindings", [])
        except Exception as exc:
            wait = (attempt + 1) * 5
            logger.warning("SPARQL attempt %d failed: %s — retrying in %ds", attempt+1, exc, wait)
            time.sleep(wait)
    return []


def add_row(g: Graph, row: dict, subj_key: str, prop_key: str, val_key: str,
            label_key: str | None = None) -> int:
    """Parse one SPARQL result row and add triple(s) to graph. Returns count added."""
    added = 0
    try:
        s_str = row.get(subj_key, {}).get("value", "")
        p_str = row.get(prop_key, {}).get("value", "")
        v     = row.get(val_key, {})
        if not s_str or not p_str or not v:
            return 0
        s = URIRef(s_str)
        p = URIRef(p_str)
        if v.get("type") == "uri":
            g.add((s, p, URIRef(v["value"]))); added += 1
        else:
            lang  = v.get("xml:lang")
            dtype = v.get("datatype")
            if lang:
                g.add((s, p, Literal(v["value"], lang=lang))); added += 1
            elif dtype:
                g.add((s, p, Literal(v["value"], datatype=URIRef(dtype)))); added += 1
            else:
                g.add((s, p, Literal(v["value"]))); added += 1
        if label_key:
            lbl = row.get(label_key, {}).get("value", "")
            if lbl and v.get("type") == "uri":
                g.add((URIRef(v["value"]), RDFS.label, Literal(lbl, lang="en")))
    except Exception:
        pass
    return added


# ---------------------------------------------------------------------------
# Broad jazz-domain queries  (the bulk of the 50k triples)
# ---------------------------------------------------------------------------

def fetch_jazz_musicians(g: Graph, session: requests.Session) -> None:
    """Jazz musicians — split into two lighter queries to avoid Wikidata 502."""
    logger.info("Fetching jazz musicians from Wikidata…")

    # Query A: birth/death/location/gender/citizenship
    prop_sets = [
        "wdt:P569 wdt:P570 wdt:P19 wdt:P20 wdt:P21 wdt:P27",
        "wdt:P1303 wdt:P264 wdt:P737 wdt:P136 wdt:P463",
    ]
    batch_size = 300
    for props in prop_sets:
        offset = 0
        while True:
            query = f"""
SELECT ?musician ?prop ?value WHERE {{
  ?musician wdt:P106 wd:Q639669 ;
            wdt:P136 wd:Q8341 .
  VALUES ?prop {{ {props} }}
  ?musician ?prop ?value .
}}
LIMIT {batch_size} OFFSET {offset}
"""
            rows = sparql_query(query, session)
            if not rows:
                break
            for row in rows:
                add_row(g, row, "musician", "prop", "value")
            logger.info("  musicians props=[%s] offset=%d → %d triples", props[:20], offset, len(g))
            offset += batch_size
            time.sleep(REQUEST_DELAY)
            if len(rows) < batch_size or len(g) > 60_000:
                break

    # Fetch labels separately (much lighter)
    query_labels = """
SELECT ?musician ?label WHERE {
  ?musician wdt:P106 wd:Q639669 ;
            wdt:P136 wd:Q8341 ;
            rdfs:label ?label .
  FILTER(LANG(?label) = "en")
}
LIMIT 5000
"""
    rows = sparql_query(query_labels, session)
    for row in rows:
        m = row.get("musician", {}).get("value", "")
        lbl = row.get("label", {}).get("value", "")
        if m and lbl:
            g.add((URIRef(m), RDFS.label, Literal(lbl, lang="en")))
    logger.info("  After musician labels: %d triples", len(g))
    time.sleep(REQUEST_DELAY)


def fetch_jazz_albums(g: Graph, session: requests.Session) -> None:
    """Jazz albums — performer, label, date."""
    logger.info("Fetching jazz albums from Wikidata…")
    batch_size = 300
    offset = 0
    while True:
        query = f"""
SELECT ?album ?prop ?value WHERE {{
  ?album wdt:P31 wd:Q482994 ;
         wdt:P136 wd:Q8341 .
  VALUES ?prop {{ wdt:P175 wdt:P264 wdt:P577 wdt:P136 wdt:P495 }}
  ?album ?prop ?value .
}}
LIMIT {batch_size} OFFSET {offset}
"""
        rows = sparql_query(query, session)
        if not rows:
            break
        for row in rows:
            add_row(g, row, "album", "prop", "value")
        logger.info("  jazz albums offset=%d → %d triples", offset, len(g))
        offset += batch_size
        time.sleep(REQUEST_DELAY)
        if len(rows) < batch_size or len(g) > 80_000:
            break

    # Album labels
    rows = sparql_query("""
SELECT ?album ?label WHERE {
  ?album wdt:P31 wd:Q482994 ; wdt:P136 wd:Q8341 ; rdfs:label ?label .
  FILTER(LANG(?label) = "en")
} LIMIT 5000
""", session)
    for row in rows:
        a = row.get("album", {}).get("value", "")
        lbl = row.get("label", {}).get("value", "")
        if a and lbl:
            g.add((URIRef(a), RDFS.label, Literal(lbl, lang="en")))
    logger.info("  After album labels: %d triples", len(g))
    time.sleep(REQUEST_DELAY)


def fetch_jazz_labels(g: Graph, session: requests.Session) -> None:
    """Jazz record labels and their artists."""
    logger.info("Fetching jazz record labels…")
    query = """
SELECT ?label ?prop ?value ?labelName ?valueLabel WHERE {
  ?label wdt:P31 wd:Q18127 .   # instance of: record label
  OPTIONAL { ?label wdt:P136 wd:Q8341 . }
  VALUES ?prop { wdt:P740 wdt:P17 wdt:P571 wdt:P576 wdt:P18 }
  ?label ?prop ?value .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 3000
"""
    rows = sparql_query(query, session)
    for row in rows:
        add_row(g, row, "label", "prop", "value", "valueLabel")
    logger.info("  jazz labels: %d triples total", len(g))
    time.sleep(REQUEST_DELAY)

    # Also get all artists per major label
    major_labels = [
        "Q183387",  # Blue Note
        "Q183412",  # Columbia
        "Q1096397", # Prestige
        "Q1140671", # Impulse!
        "Q183400",  # Atlantic
        "Q685533",  # Verve
        "Q183399",  # Riverside Records
        "Q1195067", # ECM Records
        "Q1626612", # Savoy Records
    ]
    for lqid in major_labels:
        q = f"""
SELECT ?artist ?prop ?value ?artistLabel WHERE {{
  ?artist wdt:P264 wd:{lqid} .
  VALUES ?prop {{ wdt:P569 wdt:P570 wdt:P19 wdt:P1303 wdt:P737 wdt:P136 wdt:P21 wdt:P27 }}
  ?artist ?prop ?value .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 2000
"""
        rows = sparql_query(q, session)
        for row in rows:
            add_row(g, row, "artist", "prop", "value")
            lbl = row.get("artistLabel", {}).get("value", "")
            a   = row.get("artist", {}).get("value", "")
            if lbl and a:
                g.add((URIRef(a), RDFS.label, Literal(lbl, lang="en")))
        logger.info("  label %s artists → %d triples total", lqid, len(g))
        time.sleep(REQUEST_DELAY)


def fetch_jazz_standards(g: Graph, session: requests.Session) -> None:
    """Jazz standards (songs) with composer, performer, year."""
    logger.info("Fetching jazz standards…")
    query = """
SELECT ?song ?prop ?value WHERE {
  ?song wdt:P31 wd:Q7936T .
  VALUES ?prop { wdt:P86 wdt:P175 wdt:P577 wdt:P264 wdt:P136 }
  ?song ?prop ?value .
} LIMIT 3000
"""
    # Q7936T doesn't exist — use jazz standard Q1238720
    query = """
SELECT ?song ?prop ?value WHERE {
  { ?song wdt:P31 wd:Q1238720 . }
  UNION
  { ?song wdt:P136 wd:Q8341 ; wdt:P31 wd:Q7366 . }
  VALUES ?prop { wdt:P86 wdt:P175 wdt:P577 wdt:P264 wdt:P136 wdt:P31 }
  ?song ?prop ?value .
} LIMIT 5000
"""
    rows = sparql_query(query, session)
    for row in rows:
        add_row(g, row, "song", "prop", "value")
    logger.info("  jazz standards: %d triples total", len(g))
    time.sleep(REQUEST_DELAY)


def fetch_jazz_instruments(g: Graph, session: requests.Session) -> None:
    """Jazz instruments and musicians who play them."""
    logger.info("Fetching jazz instrument data…")
    instruments = [
        "Q8338",    # trumpet
        "Q6654",    # saxophone
        "Q5994",    # piano
        "Q9798",    # double bass
        "Q1444",    # drums
        "Q78987",   # trombone
        "Q11405",   # clarinet
        "Q5994",    # piano
        "Q46185",   # vibraphone
        "Q212",     # guitar
    ]
    for iqid in instruments:
        q = f"""
SELECT ?musician ?prop ?value ?musicianLabel WHERE {{
  ?musician wdt:P1303 wd:{iqid} ;
            wdt:P136 wd:Q8341 .
  VALUES ?prop {{ wdt:P569 wdt:P570 wdt:P19 wdt:P264 wdt:P737 wdt:P21 wdt:P27 }}
  ?musician ?prop ?value .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 1000
"""
        rows = sparql_query(q, session)
        for row in rows:
            add_row(g, row, "musician", "prop", "value")
            lbl = row.get("musicianLabel", {}).get("value", "")
            m   = row.get("musician", {}).get("value", "")
            if lbl and m:
                g.add((URIRef(m), RDFS.label, Literal(lbl, lang="en")))
                g.add((URIRef(m), WDT.P1303, URIRef(f"http://www.wikidata.org/entity/{iqid}")))
        time.sleep(REQUEST_DELAY)
    logger.info("  After instruments: %d triples total", len(g))


# ---------------------------------------------------------------------------
# 1-hop expansion on aligned QIDs (original logic, kept)
# ---------------------------------------------------------------------------

def get_qids_from_alignment(alignment_path: Path) -> list[str]:
    g = Graph()
    if alignment_path.exists():
        g.parse(str(alignment_path), format="turtle")
    qids = []
    for _, _, wd_uri in g.triples((None, OWL.sameAs, None)):
        if isinstance(wd_uri, URIRef) and "wikidata.org/entity/Q" in str(wd_uri):
            qids.append(str(wd_uri).split("/")[-1])
    hardcoded = [
        "Q93341","Q7346","Q4030","Q1779","Q103767","Q192505","Q215525",
        "Q210028","Q351695","Q128196","Q217137","Q272508","Q210070",
        "Q210025","Q216197","Q128987","Q91186","Q104358","Q185109",
        "Q633710","Q183387","Q183412","Q1096397","Q1140671","Q183400",
        "Q685533","Q34404","Q60","Q841189","Q185109","Q8341",
    ]
    all_qids = list(set(qids + hardcoded))
    logger.info("Total QIDs for 1-hop expansion: %d", len(all_qids))
    return all_qids


def expand_1hop(qids: list[str], g: Graph, session: requests.Session) -> None:
    """1-hop expansion on known QIDs."""
    logger.info("1-hop expansion on %d QIDs…", len(qids))
    pred_filter = "wdt:P31 wdt:P175 wdt:P264 wdt:P136 wdt:P19 wdt:P569 wdt:P570 wdt:P1303 wdt:P463 wdt:P737 wdt:P18 wdt:P21 wdt:P27 wdt:P495 wdt:P577 wdt:P123"
    batch_size = 15
    for i in range(0, len(qids), batch_size):
        batch  = qids[i:i+batch_size]
        values = " ".join(f"wd:{q}" for q in batch)
        query  = f"""
SELECT ?entity ?prop ?value ?valueLabel WHERE {{
  VALUES ?entity {{ {values} }}
  VALUES ?prop {{ {pred_filter} }}
  ?entity ?prop ?value .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 3000
"""
        rows = sparql_query(query, session)
        for row in rows:
            add_row(g, row, "entity", "prop", "value", "valueLabel")
        time.sleep(REQUEST_DELAY)
    logger.info("After 1-hop: %d triples", len(g))


# ---------------------------------------------------------------------------
# Merge + clean
# ---------------------------------------------------------------------------

def merge_graphs(base_path: Path, alignment_path: Path, expansion_g: Graph) -> Graph:
    merged = Graph()
    merged.bind("jazz", JAZZ)
    merged.bind("wd", WD)
    merged.bind("wdt", WDT)
    merged.bind("rdfs", RDFS)
    merged.bind("owl", OWL)

    if base_path.exists():
        base_g = Graph()
        base_g.parse(str(base_path), format="turtle")
        for triple in base_g:
            merged.add(triple)
        logger.info("Merged base KG: %d triples", len(base_g))

    if alignment_path.exists():
        align_g = Graph()
        align_g.parse(str(alignment_path), format="turtle")
        for triple in align_g:
            merged.add(triple)
        logger.info("Merged alignment: %d triples", len(align_g))

    before = len(merged)
    for triple in expansion_g:
        merged.add(triple)
    logger.info("Added %d expansion triples (total: %d)", len(merged) - before, len(merged))
    return merged


def clean_graph(g: Graph) -> Graph:
    to_remove = [(s, p, o) for s, p, o in g if isinstance(s, BNode)]
    for triple in to_remove:
        g.remove(triple)
    logger.info("Removed %d blank-node triples", len(to_remove))
    return g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    session = requests.Session()
    g = Graph()
    g.bind("wd", WD)
    g.bind("wdt", WDT)
    g.bind("rdfs", RDFS)

    # Broad jazz-domain queries (bulk of triples)
    fetch_jazz_musicians(g, session)
    if len(g) < 50_000:
        fetch_jazz_albums(g, session)
    if len(g) < 50_000:
        fetch_jazz_standards(g, session)
    if len(g) < 50_000:
        fetch_jazz_labels(g, session)
    if len(g) < 50_000:
        fetch_jazz_instruments(g, session)

    # 1-hop on aligned QIDs
    qids = get_qids_from_alignment(ALIGNMENT_PATH)
    expand_1hop(qids, g, session)

    logger.info("Total expansion triples before merge: %d", len(g))

    merged = merge_graphs(INITIAL_KG_PATH, ALIGNMENT_PATH, g)
    merged = clean_graph(merged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged.serialize(destination=str(OUTPUT_PATH), format="nt")

    logger.info("Expanded KB saved → %s  (%d triples)", OUTPUT_PATH, len(merged))

    if len(merged) < 50_000:
        logger.warning("Only %d triples (target ≥50k). Wikidata may have throttled queries.", len(merged))
    else:
        logger.info("✓ Target met: %d triples", len(merged))


if __name__ == "__main__":
    main()
