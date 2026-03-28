"""
Orchestrator — Jazz Knowledge Graph Pipeline
=============================================
Coordinates all 6 pipeline steps in order:
  1. Web crawler        → data/crawler_output.jsonl
  2. NER extraction     → data/extracted_knowledge.csv
  3. Ontology build     → kg_artifacts/ontology.ttl
  4. KG construction    → kg_artifacts/initial_kg.ttl
  5. Entity alignment   → kg_artifacts/alignment.ttl
  6. KB expansion       → kg_artifacts/expanded.nt
  7. KGE training       → kge_artifacts/
  8. RAG demo           (in-memory)
  9. Reasoning          (in-memory, adds inferred triples)

Run:
    python src/orchestrator/main.py
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — makes all sibling packages importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

for _dir in (str(SRC_ROOT / "crawl"), str(SRC_ROOT / "ie"),
             str(SRC_ROOT / "kg"), str(SRC_ROOT / "kge"),
             str(SRC_ROOT / "rag"), str(SRC_ROOT / "reason"),
             str(SRC_ROOT / "orchestrator")):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# ---------------------------------------------------------------------------
# Imports from pipeline modules
# ---------------------------------------------------------------------------

from rdflib import Graph

# kg modules
from ontology import build_ontology, save_ontology                    # noqa: E402
from build_kg import build_initial_kg, add_jazz_facts, remove_isolated_nodes, load_csv  # noqa: E402
from alignment import build_alignment                                  # noqa: E402
from expand_kb import get_qids_from_alignment, expand_from_qids, merge_graphs, clean_graph  # noqa: E402

# optional / implemented modules
from kge_model import KGEModel                                         # noqa: E402
from rag_pipeline import RAGPipeline                                   # noqa: E402
from reasoner import Reasoner                                          # noqa: E402
from pipeline_validator import validate_all, print_report              # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
KG_DIR = PROJECT_ROOT / "kg_artifacts"
KGE_DIR = PROJECT_ROOT / "kge_artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
KG_DIR.mkdir(parents=True, exist_ok=True)
KGE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------

class JazzKGPipelineOrchestrator:
    """Runs and validates the full Jazz KG pipeline."""

    def __init__(self):
        logger.info("Initialising Jazz KG Pipeline Orchestrator.")
        self.kg: Graph = Graph()

        # File paths
        self.crawler_output_path = DATA_DIR / "crawler_output.jsonl"
        self.extracted_knowledge_path = DATA_DIR / "extracted_knowledge.csv"
        self.ontology_path = KG_DIR / "ontology.ttl"
        self.initial_kg_path = KG_DIR / "initial_kg.ttl"
        self.alignment_path = KG_DIR / "alignment.ttl"
        self.expanded_path = KG_DIR / "expanded.nt"

        logger.info("Orchestrator ready.")

    # ------------------------------------------------------------------ #
    # Step 1: Crawler                                                     #
    # ------------------------------------------------------------------ #

    def run_crawler(self) -> bool:
        """
        Run the Wikipedia crawler.  If output already exists (and has ≥20 pages)
        we skip re-crawling to save time.
        """
        logger.info("=== Step 1: Web Crawler ===")
        if self.crawler_output_path.exists():
            # Count lines quickly
            with open(self.crawler_output_path, "r", encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            if n >= 20:
                logger.info("Crawler output already present (%d pages). Skipping.", n)
                return True

        try:
            # Import lazily — spaCy may not be installed
            sys.path.insert(0, str(SRC_ROOT / "crawl"))
            from crawler import JazzCrawler  # type: ignore
            crawler = JazzCrawler()
            crawler.run()
            logger.info("Crawler finished.")
            return True
        except ImportError as exc:
            logger.error("Cannot import crawler: %s", exc)
            return False
        except Exception as exc:
            logger.error("Crawler failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Step 2: NER pipeline                                                #
    # ------------------------------------------------------------------ #

    def run_information_extraction(self) -> bool:
        """Run spaCy NER + relation extraction."""
        logger.info("=== Step 2: Information Extraction (NER) ===")
        if self.extracted_knowledge_path.exists():
            import csv as _csv
            with open(self.extracted_knowledge_path, encoding="utf-8") as f:
                n = sum(1 for _ in _csv.DictReader(f))
            if n >= 100:
                logger.info("extracted_knowledge.csv already present (%d rows). Skipping.", n)
                return True

        try:
            sys.path.insert(0, str(SRC_ROOT / "ie"))
            from ner_pipeline import NERPipeline  # type: ignore
            pipeline = NERPipeline()
            pipeline.run()
            logger.info("NER pipeline finished.")
            return True
        except ImportError as exc:
            logger.error("Cannot import NER pipeline: %s", exc)
            return False
        except Exception as exc:
            logger.error("NER pipeline failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Step 3: Ontology                                                    #
    # ------------------------------------------------------------------ #

    def build_ontology_step(self) -> bool:
        """Build and save the OWL ontology."""
        logger.info("=== Step 3: Ontology ===")
        try:
            g = build_ontology()
            save_ontology(g, self.ontology_path)
            logger.info("Ontology saved (%d triples).", len(g))
            return True
        except Exception as exc:
            logger.error("Ontology build failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Step 4: Initial KG                                                  #
    # ------------------------------------------------------------------ #

    def build_knowledge_graph(self) -> bool:
        """Build the initial RDF KG from extracted CSV + curated facts."""
        logger.info("=== Step 4: Build Initial KG ===")
        try:
            rows = load_csv(self.extracted_knowledge_path)
            self.kg = build_initial_kg(rows)
            self.kg = add_jazz_facts(self.kg)
            removed = remove_isolated_nodes(self.kg)
            self.initial_kg_path.parent.mkdir(parents=True, exist_ok=True)
            self.kg.serialize(destination=str(self.initial_kg_path), format="turtle")
            logger.info(
                "Initial KG saved → %s  (%d triples, removed %d isolated nodes).",
                self.initial_kg_path, len(self.kg), removed
            )
            return True
        except Exception as exc:
            logger.error("KG build failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Step 5: Entity alignment                                            #
    # ------------------------------------------------------------------ #

    def run_alignment(self) -> bool:
        """Align KG entities with Wikidata via wbsearchentities API."""
        logger.info("=== Step 5: Entity Alignment ===")
        try:
            g_align = build_alignment(
                kg_path=self.initial_kg_path,
                output_path=self.alignment_path,
            )
            logger.info("Alignment complete (%d triples).", len(g_align))
            return True
        except Exception as exc:
            logger.error("Alignment failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Step 6: KB expansion                                                #
    # ------------------------------------------------------------------ #

    def run_expansion(self) -> bool:
        """Expand the KG via SPARQL queries to Wikidata."""
        logger.info("=== Step 6: KB Expansion (SPARQL) ===")
        try:
            import requests as _req
            session = _req.Session()
            qids = get_qids_from_alignment(self.alignment_path)
            expansion_g = expand_from_qids(qids, session)
            merged = merge_graphs(self.initial_kg_path, self.alignment_path, expansion_g)
            merged = clean_graph(merged)
            self.expanded_path.parent.mkdir(parents=True, exist_ok=True)
            merged.serialize(destination=str(self.expanded_path), format="nt")
            logger.info("Expanded KB saved → %s  (%d triples).", self.expanded_path, len(merged))
            if len(merged) < 50_000:
                logger.warning(
                    "Expanded KB has %d triples (target ≥50k). "
                    "Network access required for full SPARQL expansion.", len(merged)
                )
            # Update self.kg to the merged graph for downstream steps
            self.kg = merged
            return True
        except Exception as exc:
            logger.error("Expansion failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Step 7: KGE                                                         #
    # ------------------------------------------------------------------ #

    def run_kge(self) -> bool:
        """Train TransE embeddings on the KG."""
        logger.info("=== Step 7: Knowledge Graph Embeddings ===")
        try:
            # Load the best available KG
            g = self._load_best_kg()
            kge = KGEModel(g, KGE_DIR, dim=50, n_epochs=20)
            kge.train_embeddings()
            emb = kge.get_embeddings()
            logger.info(
                "KGE done: %d entity embeddings, %d relation embeddings.",
                len(emb["entities"]), len(emb["relations"])
            )
            return True
        except Exception as exc:
            logger.error("KGE failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Step 8: RAG demo                                                    #
    # ------------------------------------------------------------------ #

    def run_rag(self) -> bool:
        """Demonstrate RAG queries against the KG."""
        logger.info("=== Step 8: RAG Pipeline ===")
        try:
            g = self._load_best_kg()
            rag = RAGPipeline(g)
            examples = [
                ("Who is Miles Davis and what instrument did he play?",
                 "Provide a concise biography."),
                ("Tell me about Blue Note Records",
                 "Describe this record label."),
            ]
            for query, prompt in examples:
                response = rag.run_rag_pipeline(query, prompt)
                logger.info("RAG query: %r", query)
                logger.info("RAG response:\n%s", response)
            return True
        except Exception as exc:
            logger.error("RAG failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Step 9: Reasoning                                                   #
    # ------------------------------------------------------------------ #

    def run_reasoning(self) -> bool:
        """Apply SWRL-like inference rules and validate KG consistency."""
        logger.info("=== Step 9: Reasoning ===")
        try:
            g = self._load_best_kg()
            reasoner = Reasoner(g)
            inferred = reasoner.infer_new_facts()
            consistent = reasoner.validate_consistency()
            logger.info(
                "Reasoning complete: %d new facts inferred. Consistent: %s.",
                inferred, consistent
            )
            return True
        except Exception as exc:
            logger.error("Reasoning failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate_pipeline_output(self) -> bool:
        """Run the full lab contract validation and print a report."""
        logger.info("=== Validating Pipeline Outputs ===")
        results = validate_all(PROJECT_ROOT)
        print_report(results)
        return all(v["passed"] for v in results.values())

    # ------------------------------------------------------------------ #
    # Full pipeline                                                       #
    # ------------------------------------------------------------------ #

    def run_full_pipeline(self) -> bool:
        """Execute all pipeline steps in order, then validate."""
        logger.info("Starting full Jazz KG pipeline.")

        steps = [
            ("Crawler",             self.run_crawler),
            ("NER",                 self.run_information_extraction),
            ("Ontology",            self.build_ontology_step),
            ("Build KG",            self.build_knowledge_graph),
            ("Alignment",           self.run_alignment),
            ("Expansion",           self.run_expansion),
            ("KGE",                 self.run_kge),
            ("RAG",                 self.run_rag),
            ("Reasoning",           self.run_reasoning),
        ]

        for step_name, step_fn in steps:
            success = step_fn()
            if not success:
                logger.error("Pipeline stopped at step: %s", step_name)
                # Validate whatever we have so far
                self.validate_pipeline_output()
                return False

        logger.info("All pipeline steps completed successfully.")
        return self.validate_pipeline_output()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_best_kg(self) -> Graph:
        """Load the richest available KG artifact."""
        g = Graph()
        if self.expanded_path.exists():
            g.parse(str(self.expanded_path), format="nt")
            logger.info("Loaded expanded KG (%d triples).", len(g))
        elif self.initial_kg_path.exists():
            g.parse(str(self.initial_kg_path), format="turtle")
            logger.info("Loaded initial KG (%d triples).", len(g))
        elif len(self.kg) > 0:
            g = self.kg
            logger.info("Using in-memory KG (%d triples).", len(g))
        else:
            logger.warning("No KG available; using empty graph.")
        return g


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    orchestrator = JazzKGPipelineOrchestrator()
    success = orchestrator.run_full_pipeline()
    if success:
        print("\nJazz KG Pipeline ran successfully and passed all validation checks.")
    else:
        print("\nJazz KG Pipeline finished with errors or failed validation. Check logs above.")
    sys.exit(0 if success else 1)
