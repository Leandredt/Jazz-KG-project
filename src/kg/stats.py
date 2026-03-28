"""
Lab 2 - KB Statistics
======================
Generates a statistics report about the Knowledge Base.

Usage:
    python src/kg/stats.py

Output:
    Prints statistics to stdout.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
from collections import Counter

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef

from ontology import JAZZ

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kb_stats")


def compute_stats(g: Graph) -> dict:
    """Compute statistics for an RDF graph."""
    stats = {}

    # Basic counts
    stats["total_triples"] = len(g)
    stats["unique_subjects"] = len(set(g.subjects()))
    stats["unique_predicates"] = len(set(g.predicates()))
    stats["unique_objects"] = len(set(g.objects()))

    # Entity type counts
    type_counts = Counter()
    for _, _, cls in g.triples((None, RDF.type, None)):
        type_counts[str(cls)] = type_counts.get(str(cls), 0) + 1
    stats["entity_types"] = dict(type_counts)

    # Jazz class counts
    jazz_classes = [
        ("Musician", JAZZ.Musician),
        ("Album", JAZZ.Album),
        ("RecordLabel", JAZZ.RecordLabel),
        ("Band", JAZZ.Band),
        ("Location", JAZZ.Location),
        ("Genre", JAZZ.Genre),
        ("Instrument", JAZZ.Instrument),
    ]
    for name, cls in jazz_classes:
        count = sum(1 for _ in g.subjects(RDF.type, cls))
        stats[f"jazz_{name.lower()}_count"] = count

    # Property usage counts
    prop_counts = Counter()
    for s, p, o in g:
        if p != RDF.type and p != RDFS.label:
            prop_counts[str(p)] += 1
    stats["top_predicates"] = dict(prop_counts.most_common(20))

    # Alignment stats
    same_as_count = sum(1 for _ in g.triples((None, OWL.sameAs, None)))
    stats["aligned_entities"] = same_as_count

    # Isolated nodes (have only rdf:type)
    connected_subjects = set()
    for s, p, o in g:
        if p != RDF.type and p != RDFS.label:
            if isinstance(s, URIRef):
                connected_subjects.add(s)
            if isinstance(o, URIRef):
                connected_subjects.add(o)
    all_jazz_subjects = set(
        s for s in g.subjects(RDF.type, None)
        if isinstance(s, URIRef) and "jazz-kg.org" in str(s)
    )
    stats["isolated_nodes"] = len(all_jazz_subjects - connected_subjects)
    stats["connected_jazz_entities"] = len(all_jazz_subjects & connected_subjects)

    return stats


def print_stats(stats: dict) -> None:
    print("\n" + "=" * 60)
    print("JAZZ KNOWLEDGE BASE STATISTICS")
    print("=" * 60)
    print(f"Total triples:          {stats['total_triples']:,}")
    print(f"Unique subjects:        {stats['unique_subjects']:,}")
    print(f"Unique predicates:      {stats['unique_predicates']:,}")
    print(f"Unique objects:         {stats['unique_objects']:,}")
    print(f"Aligned entities:       {stats['aligned_entities']:,}")
    print(f"Isolated nodes:         {stats['isolated_nodes']:,}")
    print(f"Connected Jazz entities:{stats['connected_jazz_entities']:,}")
    print()
    print("Jazz Entity Counts:")
    for key, val in stats.items():
        if key.startswith("jazz_") and key.endswith("_count"):
            name = key.replace("jazz_", "").replace("_count", "").capitalize()
            print(f"  {name:20s}: {val:,}")
    print()
    print("Top 10 Predicates by Usage:")
    for pred, count in list(stats.get("top_predicates", {}).items())[:10]:
        short_pred = pred.split("/")[-1].split("#")[-1]
        print(f"  {short_pred:30s}: {count:,}")
    print("=" * 60)


def main() -> None:
    for name, path_suffix, fmt in [
        ("Initial KG", "kg_artifacts/initial_kg.ttl", "turtle"),
        ("Alignment", "kg_artifacts/alignment.ttl", "turtle"),
        ("Expanded KB", "kg_artifacts/expanded.nt", "nt"),
    ]:
        path = PROJECT_ROOT / path_suffix
        if not path.exists():
            print(f"\n[MISSING] {name}: {path}")
            continue
        print(f"\nLoading {name} from {path}...")
        g = Graph()
        g.parse(str(path), format=fmt)
        stats = compute_stats(g)
        print(f"  [{name}]")
        print_stats(stats)


if __name__ == "__main__":
    main()
