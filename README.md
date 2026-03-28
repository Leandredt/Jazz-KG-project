# Jazz Knowledge Graph

End-to-end Knowledge Graph pipeline for Jazz music built from Wikipedia + Wikidata.

## Requirements

- Python 3.12 (for the UI and visualization)
- Python 3.x (for the pipeline)

## Installation

### 1. Pipeline dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_trf
```

### 2. UI / visualization dependencies (requires Python 3.12)

```bash
python3.12 -m venv venv_ui
source venv_ui/bin/activate
pip install -r requirements_ui.txt
```

## Run the full pipeline

```bash
source venv/bin/activate
bash run_pipeline.sh
```

Or step by step with `make`:

```bash
make crawl   # Step 1 — crawl Wikipedia
make ner     # Step 2 — extract entities
make kg      # Step 3-5 — build ontology, KG, alignment
make expand  # Step 6 — expand with Wikidata SPARQL
```

## Run tests

```bash
source venv/bin/activate
make test
```

## Launch the Streamlit interface

```bash
source venv_ui/bin/activate
streamlit run src/app/streamlit_app.py
# → http://localhost:8501
```

## Launch the Jupyter notebook

```bash
source venv_ui/bin/activate
jupyter notebook notebooks/jazz_kg_exploration.ipynb
```

## Project structure

```
jazz-kg-project/
├── src/
│   ├── crawl/          # Wikipedia crawler
│   ├── ie/             # NER pipeline (spaCy)
│   ├── kg/             # Ontology, KG builder, alignment, expansion
│   ├── kge/            # Knowledge Graph Embeddings (TransE)
│   ├── reason/         # OWL reasoning
│   ├── rag/            # RAG pipeline
│   ├── app/            # Streamlit interface
│   └── orchestrator/   # Pipeline orchestrator + validator
├── data/               # Generated data (gitignored)
├── kg_artifacts/       # Generated KG files (gitignored)
├── notebooks/          # Jupyter exploration notebook
├── reports/            # Generated charts and HTML graphs
├── tests/              # pytest test suite
├── requirements.txt         # Pipeline dependencies
├── requirements_ui.txt      # UI/visualization dependencies
├── Makefile
└── run_pipeline.sh
```
