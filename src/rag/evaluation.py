"""
RAG Evaluation
==============
Compares keyword-based baseline (rag_pipeline.py) against the NL→SPARQL
pipeline (nl_sparql.py) on ≥5 test questions.

Saves results to reports/rag_evaluation.json and prints a comparison table.

Usage:
    python src/rag/evaluation.py
"""

import json
import logging
import sys
import time
from pathlib import Path

from rdflib import Graph

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.rag_pipeline import RAGPipeline
from rag.nl_sparql import NLToSPARQL

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    "Who are the musicians that play trumpet?",
    "What albums did Miles Davis record?",
    "Which record labels are associated with bebop music?",
    "Who influenced John Coltrane?",
    "What instruments are played in jazz fusion?",
    "Which musicians were born in New Orleans?",
    "What are the albums released by Blue Note Records?",
]

# ---------------------------------------------------------------------------
# KG loading
# ---------------------------------------------------------------------------

def load_kg() -> Graph:
    g = Graph()
    for fname, fmt in [
        ("expanded.nt", "nt"),
        ("initial_kg.ttl", "turtle"),
        ("ontology.ttl", "turtle"),
    ]:
        p = PROJECT_ROOT / "kg_artifacts" / fname
        if p.exists():
            print(f"  Loading {fname} ...", end=" ", flush=True)
            g.parse(str(p), format=fmt)
            print("ok")
    print(f"  Total triples: {len(g):,}")
    return g

# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def run_baseline(rag: RAGPipeline, question: str) -> dict:
    t0 = time.time()
    context = rag.retrieve_info(question)
    elapsed = time.time() - t0

    # Extract top-3 entity summaries
    lines = context.splitlines()
    top3 = lines[:3] if lines else []

    return {
        "answer": context,
        "top3": top3,
        "elapsed_s": round(elapsed, 3),
    }

# ---------------------------------------------------------------------------
# NL→SPARQL evaluation
# ---------------------------------------------------------------------------

def run_nl_sparql(nl: NLToSPARQL, question: str) -> dict:
    t0 = time.time()
    result = nl.answer(question)
    elapsed = time.time() - t0
    result["elapsed_s"] = round(elapsed, 3)
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Jazz KG — RAG Evaluation: Baseline vs NL→SPARQL")
    print("=" * 70)
    print()

    # Load KG
    print("Loading Knowledge Graph...")
    g = load_kg()
    print()

    # Init pipelines
    print("Initialising pipelines...")
    baseline_rag = RAGPipeline(g, llm_model_name="kg-rag")

    ollama_available = True
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        # Pick best available model
        tags = resp.json().get("models", [])
        model_names = [m["name"] for m in tags]
        preferred = ["mistral:7b", "llama3.1:8b", "qwen2.5:7b"]
        chosen_model = next((m for m in preferred if m in model_names), None)
        if chosen_model is None and model_names:
            chosen_model = model_names[0]
        elif chosen_model is None:
            chosen_model = "mistral:7b"
        print(f"  Ollama is running. Using model: {chosen_model}")
    except Exception as exc:
        print(f"  WARNING: Ollama not available ({exc}). NL→SPARQL will report errors.")
        ollama_available = False
        chosen_model = "mistral:7b"

    nl_sparql = NLToSPARQL(g, model=chosen_model)
    print()

    # Run evaluations
    all_results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")

        # Baseline
        print("  → Baseline (keyword search) ...", end=" ", flush=True)
        baseline = run_baseline(baseline_rag, question)
        print(f"done ({baseline['elapsed_s']}s)")

        # NL→SPARQL
        print("  → NL→SPARQL pipeline ...", end=" ", flush=True)
        nl_result = run_nl_sparql(nl_sparql, question)
        print(f"done ({nl_result['elapsed_s']}s)")

        entry = {
            "question": question,
            "baseline": {
                "answer_preview": baseline["top3"],
                "elapsed_s": baseline["elapsed_s"],
            },
            "nl_sparql": {
                "sparql_query": nl_result.get("sparql"),
                "results_count": len(nl_result.get("results", [])),
                "results_preview": nl_result.get("results", [])[:3],
                "success": nl_result.get("success", False),
                "repairs_needed": nl_result.get("repairs_needed", 0),
                "error": nl_result.get("error"),
                "answer_text": nl_result.get("answer_text", ""),
                "elapsed_s": nl_result["elapsed_s"],
            },
        }
        all_results.append(entry)
        print()

    # Save JSON report
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "rag_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {report_path}")
    print()

    # Print comparison table
    _print_table(all_results)


def _print_table(results: list):
    col_w = [4, 42, 10, 10, 10, 10, 10]
    header = ["#", "Question", "Base (s)", "NL (s)", "Executed", "Repairs", "Results"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

    def row_str(cells):
        parts = []
        for cell, w in zip(cells, col_w):
            s = str(cell)
            if len(s) > w:
                s = s[: w - 1] + "…"
            parts.append(f" {s:<{w}} ")
        return "|" + "|".join(parts) + "|"

    print("=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(sep)
    print(row_str(header))
    print(sep)

    for i, entry in enumerate(results, 1):
        q = entry["question"]
        b_t = entry["baseline"]["elapsed_s"]
        nl = entry["nl_sparql"]
        nl_t = nl["elapsed_s"]
        executed = "YES" if nl["success"] else "NO"
        repairs = nl["repairs_needed"]
        n_results = nl["results_count"]
        print(row_str([i, q, b_t, nl_t, executed, repairs, n_results]))

    print(sep)
    print()

    # Summary statistics
    total = len(results)
    succeeded = sum(1 for r in results if r["nl_sparql"]["success"])
    repaired = sum(1 for r in results if r["nl_sparql"]["repairs_needed"] > 0)

    print(f"Summary:")
    print(f"  Questions tested  : {total}")
    print(f"  SPARQL succeeded  : {succeeded}/{total}")
    print(f"  Self-repair used  : {repaired}/{total}")
    print()

    # Per-question detail
    print("=" * 70)
    print("DETAILED ANSWERS")
    print("=" * 70)
    for i, entry in enumerate(results, 1):
        print(f"\n[{i}] {entry['question']}")
        print("-" * 60)

        print("  BASELINE (top 3 keyword hits):")
        preview = entry["baseline"]["answer_preview"]
        if preview:
            for line in preview:
                print(f"    {line[:100]}")
        else:
            print("    (no results)")

        print()
        nl = entry["nl_sparql"]
        print("  NL→SPARQL:")
        if nl["sparql_query"]:
            # Show first 6 lines of the SPARQL
            sparql_lines = nl["sparql_query"].splitlines()[:6]
            for line in sparql_lines:
                print(f"    {line}")
            if len(nl["sparql_query"].splitlines()) > 6:
                print("    ...")
        print(f"  Success: {nl['success']}  |  Repairs: {nl['repairs_needed']}  |  Results: {nl['results_count']}")
        if nl["error"] and not nl["success"]:
            print(f"  Error: {nl['error'][:120]}")
        if nl["results_preview"]:
            print("  Top results:")
            for row in nl["results_preview"]:
                vals = [f"{k}={v}" for k, v in row.items() if v]
                print(f"    • {' | '.join(vals)}")


if __name__ == "__main__":
    main()
