"""
KGE Data Preparation
====================
Loads expanded.nt, filters URI-only triples, builds entity/relation mappings,
and splits into train (80%) / valid (10%) / test (10%) in data/kge/.
"""

import random
from pathlib import Path

from rdflib import Graph, URIRef

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KG_PATH = PROJECT_ROOT / "kg_artifacts" / "expanded.nt"
OUT_DIR = PROJECT_ROOT / "data" / "kge"


def prepare(kg_path: Path = KG_PATH, out_dir: Path = OUT_DIR, seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading KG from {kg_path} …")
    g = Graph()
    g.parse(str(kg_path), format="nt")
    print(f"  Loaded {len(g)} total triples.")

    # Filter URI-only triples
    triples = []
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
            triples.append((str(s), str(p), str(o)))

    print(f"  URI-only triples: {len(triples)}")

    # Build mappings
    entities: set[str] = set()
    relations: set[str] = set()
    for s, p, o in triples:
        entities.add(s)
        entities.add(o)
        relations.add(p)

    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}

    # Save mappings (tab-separated: uri \t id)
    with open(out_dir / "entity2id.txt", "w", encoding="utf-8") as f:
        f.write(f"{len(entity2id)}\n")
        for uri, idx in entity2id.items():
            f.write(f"{uri}\t{idx}\n")

    with open(out_dir / "relation2id.txt", "w", encoding="utf-8") as f:
        f.write(f"{len(relation2id)}\n")
        for uri, idx in relation2id.items():
            f.write(f"{uri}\t{idx}\n")

    print(f"  Entities: {len(entity2id)}, Relations: {len(relation2id)}")

    # Convert to id triples
    id_triples = [
        (entity2id[s], relation2id[p], entity2id[o])
        for s, p, o in triples
    ]

    # Shuffle and split
    random.seed(seed)
    random.shuffle(id_triples)
    n = len(id_triples)
    n_train = max(1, int(n * 0.8))
    n_valid = max(1, int(n * 0.1))
    train = id_triples[:n_train]
    valid = id_triples[n_train: n_train + n_valid]
    test = id_triples[n_train + n_valid:]

    # Write splits (tab-separated: head \t relation \t tail)
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        path = out_dir / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in split:
                f.write(f"{h}\t{r}\t{t}\n")
        print(f"  {name}.txt: {len(split)} triples → {path}")

    # Also write the same splits to data/ root (for grading)
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        path = PROJECT_ROOT / "data" / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in split:
                f.write(f"{h}\t{r}\t{t}\n")

    return {
        "n_entities": len(entity2id),
        "n_relations": len(relation2id),
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "entity2id": entity2id,
        "relation2id": relation2id,
        "train": train,
        "valid": valid,
        "test": test,
    }


if __name__ == "__main__":
    info = prepare()
    print("\nSummary:")
    print(f"  Entities  : {info['n_entities']}")
    print(f"  Relations : {info['n_relations']}")
    print(f"  Train     : {info['n_train']}")
    print(f"  Valid     : {info['n_valid']}")
    print(f"  Test      : {info['n_test']}")
