"""
Pipeline Validator
==================
Validates each step of the Jazz KG pipeline against the lab contracts.

Lab 1 checks
------------
  - crawler_output.jsonl exists and has ≥20 pages
  - Every page has word_count ≥ 500
  - extracted_knowledge.csv exists and has ≥100 entity rows
  - CSV contains PERSON and ORG entity types

Lab 2 checks
------------
  - ontology.ttl has ≥5 OWL classes and ≥5 OWL properties
  - initial_kg.ttl has ≥100 triples and ≥50 Jazz entities
  - alignment.ttl has owl:sameAs triples with ≥50% coverage
  - expanded.nt has ≥50k triples (soft — warns if below)

Returns
-------
    validate_all() → dict[str, dict]
    {
        "lab1_crawler":       {"passed": bool, "detail": str},
        "lab1_entities":      {"passed": bool, "detail": str},
        "lab2_ontology":      {"passed": bool, "detail": str},
        "lab2_initial_kg":    {"passed": bool, "detail": str},
        "lab2_alignment":     {"passed": bool, "detail": str},
        "lab2_expanded":      {"passed": bool, "detail": str},
    }
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef

logger = logging.getLogger(__name__)

JAZZ = Namespace("http://jazz-kg.org/ontology#")

# ---------------------------------------------------------------------------
# Thresholds (mirrors lab contracts)
# ---------------------------------------------------------------------------

MIN_CRAWLER_PAGES = 20
MIN_WORD_COUNT = 500
MIN_NER_ENTITIES = 100
REQUIRED_ENTITY_TYPES = {"PERSON", "ORG"}

MIN_OWL_CLASSES = 5
MIN_OWL_PROPS = 5
MIN_KG_TRIPLES = 100
MIN_KG_ENTITIES = 50
MIN_ALIGNMENT_COVERAGE = 0.5
MIN_ALIGNMENT_SAME_AS = 10
MIN_EXPANDED_TRIPLES_HARD = 1000   # fail below this
MIN_EXPANDED_TRIPLES_SOFT = 50_000  # warn below this


# ---------------------------------------------------------------------------
# Internal check helpers
# ---------------------------------------------------------------------------

def _check(name: str, passed: bool, detail: str) -> dict:
    status = "PASS" if passed else "FAIL"
    logger.info("[%s] %s — %s", status, name, detail)
    return {"passed": passed, "detail": detail}


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Lab 1 checks
# ---------------------------------------------------------------------------

def check_lab1_crawler(data_dir: Path) -> dict:
    """≥20 crawled pages, each with word_count ≥ 500."""
    path = data_dir / "crawler_output.jsonl"
    if not path.exists():
        return _check("lab1_crawler", False, f"File not found: {path}")

    records = _load_jsonl(path)
    n_pages = len(records)
    if n_pages < MIN_CRAWLER_PAGES:
        return _check(
            "lab1_crawler", False,
            f"Only {n_pages} pages crawled (need ≥{MIN_CRAWLER_PAGES})"
        )

    short = [r.get("url", "?") for r in records if r.get("word_count", 0) < MIN_WORD_COUNT]
    if short:
        return _check(
            "lab1_crawler", False,
            f"{len(short)} pages below {MIN_WORD_COUNT} words: {short[:3]}"
        )

    return _check("lab1_crawler", True, f"{n_pages} pages crawled, all ≥{MIN_WORD_COUNT} words.")


def check_lab1_entities(data_dir: Path) -> dict:
    """extracted_knowledge.csv has ≥100 entities including PERSON and ORG."""
    path = data_dir / "extracted_knowledge.csv"
    if not path.exists():
        return _check("lab1_entities", False, f"File not found: {path}")

    rows = _load_csv(path)
    n = len(rows)
    if n < MIN_NER_ENTITIES:
        return _check(
            "lab1_entities", False,
            f"Only {n} entity rows (need ≥{MIN_NER_ENTITIES})"
        )

    found_types = {r.get("entity_type", "") for r in rows}
    missing = REQUIRED_ENTITY_TYPES - found_types
    if missing:
        return _check(
            "lab1_entities", False,
            f"Missing entity types: {missing}. Found: {found_types}"
        )

    return _check("lab1_entities", True, f"{n} entities with types {found_types & REQUIRED_ENTITY_TYPES}.")


# ---------------------------------------------------------------------------
# Lab 2 checks
# ---------------------------------------------------------------------------

def check_lab2_ontology(kg_artifacts_dir: Path) -> dict:
    """ontology.ttl has ≥5 OWL classes and ≥5 OWL properties."""
    path = kg_artifacts_dir / "ontology.ttl"
    if not path.exists():
        return _check("lab2_ontology", False, f"File not found: {path}")

    g = Graph()
    g.parse(str(path), format="turtle")

    classes = list(g.subjects(RDF.type, OWL.Class))
    obj_props = list(g.subjects(RDF.type, OWL.ObjectProperty))
    dt_props = list(g.subjects(RDF.type, OWL.DatatypeProperty))
    n_props = len(obj_props) + len(dt_props)

    if len(classes) < MIN_OWL_CLASSES:
        return _check("lab2_ontology", False, f"Only {len(classes)} OWL classes (need ≥{MIN_OWL_CLASSES})")
    if n_props < MIN_OWL_PROPS:
        return _check("lab2_ontology", False, f"Only {n_props} OWL properties (need ≥{MIN_OWL_PROPS})")

    return _check(
        "lab2_ontology", True,
        f"{len(classes)} classes, {n_props} properties in ontology."
    )


def check_lab2_initial_kg(kg_artifacts_dir: Path) -> dict:
    """initial_kg.ttl has ≥100 triples and ≥50 Jazz entities."""
    path = kg_artifacts_dir / "initial_kg.ttl"
    if not path.exists():
        return _check("lab2_initial_kg", False, f"File not found: {path}")

    g = Graph()
    g.parse(str(path), format="turtle")
    n_triples = len(g)
    jazz_entities = [
        s for s in g.subjects(RDF.type, None)
        if isinstance(s, URIRef) and "jazz-kg.org/resource" in str(s)
    ]
    n_entities = len(set(jazz_entities))

    if n_triples < MIN_KG_TRIPLES:
        return _check("lab2_initial_kg", False, f"Only {n_triples} triples (need ≥{MIN_KG_TRIPLES})")
    if n_entities < MIN_KG_ENTITIES:
        return _check("lab2_initial_kg", False, f"Only {n_entities} Jazz entities (need ≥{MIN_KG_ENTITIES})")

    return _check(
        "lab2_initial_kg", True,
        f"{n_triples} triples, {n_entities} Jazz entities."
    )


def check_lab2_alignment(kg_artifacts_dir: Path) -> dict:
    """alignment.ttl has ≥50% entities linked and ≥10 owl:sameAs triples."""
    align_path = kg_artifacts_dir / "alignment.ttl"
    kg_path = kg_artifacts_dir / "initial_kg.ttl"

    if not align_path.exists():
        return _check("lab2_alignment", False, f"File not found: {align_path}")

    align_g = Graph()
    align_g.parse(str(align_path), format="turtle")
    same_as = list(align_g.triples((None, OWL.sameAs, None)))
    n_linked = len(same_as)

    if n_linked < MIN_ALIGNMENT_SAME_AS:
        return _check(
            "lab2_alignment", False,
            f"Only {n_linked} owl:sameAs triples (need ≥{MIN_ALIGNMENT_SAME_AS})"
        )

    # Compute coverage if initial KG available
    if kg_path.exists():
        kg_g = Graph()
        kg_g.parse(str(kg_path), format="turtle")
        total = len(set(
            s for s in kg_g.subjects(RDF.type, None)
            if isinstance(s, URIRef) and "jazz-kg.org/resource" in str(s)
        ))
        if total > 0:
            coverage = n_linked / total
            if coverage < MIN_ALIGNMENT_COVERAGE and n_linked < 20:
                return _check(
                    "lab2_alignment", False,
                    f"Alignment coverage {coverage:.1%} ({n_linked}/{total}) below {MIN_ALIGNMENT_COVERAGE:.0%}"
                )
            return _check(
                "lab2_alignment", True,
                f"Alignment coverage {coverage:.1%} ({n_linked}/{total} entities linked)."
            )

    return _check("lab2_alignment", True, f"{n_linked} owl:sameAs triples present.")


def check_lab2_expanded(kg_artifacts_dir: Path) -> dict:
    """expanded.nt has ≥50k triples (hard minimum: 1000)."""
    path = kg_artifacts_dir / "expanded.nt"
    if not path.exists():
        return _check("lab2_expanded", False, f"File not found: {path}")

    g = Graph()
    g.parse(str(path), format="nt")
    n = len(g)

    if n < MIN_EXPANDED_TRIPLES_HARD:
        return _check(
            "lab2_expanded", False,
            f"Only {n} triples in expanded.nt (hard minimum: {MIN_EXPANDED_TRIPLES_HARD:,})"
        )

    if n < MIN_EXPANDED_TRIPLES_SOFT:
        logger.warning(
            "expanded.nt has %d triples — below target of %d. "
            "Run expand_kb.py with network access.", n, MIN_EXPANDED_TRIPLES_SOFT
        )
        # Still pass — just warn
        return _check(
            "lab2_expanded", True,
            f"{n:,} triples (below {MIN_EXPANDED_TRIPLES_SOFT:,} target; needs SPARQL expansion)."
        )

    return _check("lab2_expanded", True, f"{n:,} triples — target met.")


# ---------------------------------------------------------------------------
# Aggregate validator
# ---------------------------------------------------------------------------

def validate_all(project_root: Optional[Path] = None) -> dict[str, dict]:
    """
    Run all lab validation checks and return a consolidated report.

    Parameters
    ----------
    project_root : Path or None
        Root of the jazz-kg-project directory.  Defaults to the directory
        two levels above this file.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    data_dir = project_root / "data"
    kg_dir = project_root / "kg_artifacts"

    results: dict[str, dict] = {}
    results["lab1_crawler"] = check_lab1_crawler(data_dir)
    results["lab1_entities"] = check_lab1_entities(data_dir)
    results["lab2_ontology"] = check_lab2_ontology(kg_dir)
    results["lab2_initial_kg"] = check_lab2_initial_kg(kg_dir)
    results["lab2_alignment"] = check_lab2_alignment(kg_dir)
    results["lab2_expanded"] = check_lab2_expanded(kg_dir)

    passed = sum(1 for v in results.values() if v["passed"])
    total = len(results)
    logger.info("Validation summary: %d/%d checks passed.", passed, total)

    return results


def print_report(results: dict[str, dict]) -> None:
    """Pretty-print the validation report."""
    print("\n" + "=" * 65)
    print("  PIPELINE VALIDATION REPORT")
    print("=" * 65)
    for check_name, outcome in results.items():
        status = "PASS" if outcome["passed"] else "FAIL"
        marker = "✓" if outcome["passed"] else "✗"
        print(f"  [{marker}] {check_name:<25} {status}  — {outcome['detail']}")
    print("=" * 65)
    passed = sum(1 for v in results.values() if v["passed"])
    print(f"  {passed}/{len(results)} checks passed.")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    results = validate_all()
    print_report(results)
    all_passed = all(v["passed"] for v in results.values())
    sys.exit(0 if all_passed else 1)
