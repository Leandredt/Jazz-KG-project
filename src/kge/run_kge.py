"""
KGE Pipeline Orchestrator
=========================
Runs the full KGE pipeline in sequence:
  1. Prepare data (splits + mappings)
  2. Train TransE and DistMult (via evaluate_kge.py)
  3. Evaluate and produce metrics table
  4. Plot t-SNE visualization
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KGE_DATA_DIR = PROJECT_ROOT / "data" / "kge"


def main():
    print("=" * 60)
    print("KGE Pipeline — Jazz Knowledge Graph Embeddings")
    print("=" * 60)

    # Step 1: Prepare data
    print("\n[1/2] Preparing KGE dataset splits …")
    from prepare_kge_data import prepare
    info = prepare()
    print(f"  Done. Entities={info['n_entities']}, Relations={info['n_relations']}")
    print(f"  Train={info['n_train']}, Valid={info['n_valid']}, Test={info['n_test']}")

    # Step 2: Train + Evaluate + Plot
    print("\n[2/2] Training models, evaluating, plotting …")
    from evaluate_kge import main as eval_main
    results = eval_main()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Metrics  → {PROJECT_ROOT / 'reports' / 'kge_metrics.json'}")
    print(f"  t-SNE    → {PROJECT_ROOT / 'reports' / 'tsne_embeddings.png'}")
    print(f"  Splits   → {KGE_DATA_DIR}/{{train,valid,test}}.txt")
    print(f"  Splits   → {PROJECT_ROOT / 'data'}/{{train,valid,test}}.txt")
    print("=" * 60)


if __name__ == "__main__":
    # Make sibling modules importable
    sys.path.insert(0, str(Path(__file__).parent))
    main()
