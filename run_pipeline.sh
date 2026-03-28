#!/usr/bin/env bash
# run_pipeline.sh — One-shot Jazz KG pipeline with timestamped logs
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Jazz KG Pipeline Start ==="

log "Step 1/6: Crawling Wikipedia..."
python3 src/crawl/crawler.py
log "Step 1/6: Done."

log "Step 2/6: Extracting entities and relations (NER)..."
python3 src/ie/ner_pipeline.py
log "Step 2/6: Done."

log "Step 3/6: Building ontology..."
python3 src/kg/ontology.py
log "Step 3/6: Done."

log "Step 4/6: Building initial KG..."
python3 src/kg/build_kg.py
log "Step 4/6: Done."

log "Step 5/6: Aligning entities with Wikidata..."
python3 src/kg/alignment.py
log "Step 5/6: Done."

log "Step 6/6: Expanding KB via SPARQL..."
python3 src/kg/expand_kb.py
log "Step 6/6: Done."

log "=== Validating pipeline outputs ==="
python3 src/orchestrator/pipeline_validator.py

# --- Summary ---
echo ""
echo "========================================"
echo " Pipeline Summary"
echo "========================================"

PAGES=$(wc -l < data/crawler_output.jsonl 2>/dev/null || echo 0)
ENTITIES=$(tail -n +2 data/extracted_knowledge.csv 2>/dev/null | wc -l || echo 0)
TRIPLES=$(wc -l < kg_artifacts/expanded.nt 2>/dev/null || echo 0)

echo "  Crawled pages : $PAGES"
echo "  Extracted entities : $ENTITIES"
echo "  Expanded triples : $TRIPLES"
echo "========================================"
log "=== Pipeline completed successfully ==="
