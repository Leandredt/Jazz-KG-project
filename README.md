# Jazz Knowledge Graph

End-to-end pipeline that builds a Knowledge Graph for Jazz music from Wikipedia and Wikidata, with a natural language query interface and knowledge graph embeddings.

## Overview

```
Wikipedia articles
    → Crawl & extract text         (httpx + trafilatura, 120 pages)
    → Named Entity Recognition     (spaCy en_core_web_trf, 600+ entities)
    → Ontology + RDF triples       (7 classes, 10+ properties, OWL)
    → Wikidata alignment           (owl:sameAs, ≥50% coverage)
    → SPARQL expansion             (~50,000 triples)
    → OWL reasoning                (4 inference rules)
    → KG Embeddings                (TransE MRR=0.10, DistMult MRR=0.08)
    → Streamlit app                (NL→SPARQL, graph explorer, stats)
```

## Streamlit Application

Three tabs:
- **RAG Demo** — ask a question in natural language, get an answer from the KG
- **Overview & Stats** — triple counts, entity distribution, KGE metrics, t-SNE embeddings
- **Graph Explorer** — interactive network visualization

```bash
source venv_ui/bin/activate
streamlit run src/app/streamlit_app.py
# → http://localhost:8501
```

> Requires [Ollama](https://ollama.com) running locally with Mistral 7B for the RAG tab:
> `ollama serve` then `ollama pull mistral`

## Run the Full Pipeline

```bash
source venv/bin/activate
bash run_pipeline.sh
```

Or step by step:

```bash
make crawl    # Step 1 — crawl Wikipedia
make ner      # Step 2 — extract entities
make kg       # Step 3–5 — ontology, KG, alignment
make expand   # Step 6 — expand with Wikidata SPARQL
```

## Installation

```bash
# Pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_trf

# UI (requires Python 3.12)
python3.12 -m venv venv_ui && source venv_ui/bin/activate
pip install -r requirements_ui.txt
```

## Tests

```bash
source venv/bin/activate && make test   # 56 tests
```

## Project Structure

```
jazz-kg-project/
├── src/
│   ├── crawl/          Wikipedia crawler (httpx + trafilatura)
│   ├── ie/             NER & relation extraction (spaCy)
│   ├── kg/             Ontology, KG builder, Wikidata alignment, SPARQL expansion
│   ├── reason/         OWL forward-chaining reasoner
│   ├── kge/            Knowledge Graph Embeddings (TransE, DistMult)
│   ├── rag/            NL→SPARQL pipeline (Mistral 7B + self-repair)
│   ├── app/            Streamlit interface
│   └── orchestrator/   Full pipeline orchestrator + validator
├── data/
│   ├── samples/        Sample data for testing
│   └── README.md
├── kg_artifacts/
│   ├── ontology.ttl    OWL ontology (7 classes, 10+ properties)
│   ├── initial_kg.ttl  Initial RDF graph
│   ├── alignment.ttl   Wikidata owl:sameAs links
│   └── expanded.nt     Expanded graph (~50k triples)
├── reports/
│   ├── kge_metrics.json
│   └── tsne_embeddings.png
├── notebooks/          Jupyter exploration notebook
├── tests/              pytest test suite (56 tests)
├── README.md
├── requirements.txt
├── requirements_ui.txt
├── Makefile
├── run_pipeline.sh
└── .gitignore
```
