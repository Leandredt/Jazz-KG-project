# Jazz Knowledge Graph: An End-to-End Pipeline for Music Information Extraction and Exploration

**Course**: Web Data Mining — ESILV A4
**Date**: March 2026

---

## 1. Introduction

This project constructs a domain-specific Knowledge Graph (KG) for Jazz music, integrating information extraction from the open web with semantic web standards, entity alignment to Wikidata, and downstream reasoning and embedding tasks. The system spans the full pipeline from raw Wikipedia text to a queryable, interactive application. The scope covers major jazz artists, albums, record labels, instruments, and geographic origins, with the explicit goal of producing a replicable, testable pipeline validated by a 56-test pytest suite.

---

## 2. Data Collection & Processing

### 2.1 Web Crawling

The crawler (`src/crawl/crawler.py`) uses `httpx` for HTTP requests and `trafilatura` for boilerplate-free text extraction. Starting from a curated set of Wikipedia seed URLs covering jazz artists, albums, and record labels, the crawler performs a breadth-first traversal capped at 120 pages. A 1.5-second inter-request delay and `robots.txt` compliance ensure polite, respectful harvesting. Each collected document is stored as a JSON line in `crawler_output.jsonl` with fields: `url`, `title`, `text`, and `word_count`. Quality filters enforce a minimum of 500 words per page and at least 20 pages total.

### 2.2 Named Entity Recognition & Relation Extraction

The NER pipeline (`src/ie/ner_pipeline.py`) applies spaCy's `en_core_web_trf` transformer model, which uses a BERT-class encoder for high-accuracy entity recognition. Five entity types are extracted: `PERSON`, `ORG`, `GPE`, `DATE`, and `WORK_OF_ART`. Relation extraction relies on dependency parsing: for each recognized entity pair within the same sentence, syntactic head-dependency paths determine the relation type (e.g., `nsubj`/`dobj` patterns yield `plays` or `memberOf`).

The output, `extracted_knowledge.csv`, contains **600+ entities** across six columns: `entity`, `entity_type`, `source_url`, `context_sentence`, `relation_to`, and `relation_type`. This structured format directly feeds the KG construction stage.

---

## 3. Knowledge Graph Construction

### 3.1 Ontology Design

The OWL ontology (`kg_artifacts/ontology.ttl`) defines **7 classes**: `Musician`, `Album`, `RecordLabel`, `Band`, `Location`, `Instrument`, and `Genre`. These are connected by **10+ object properties**, including `bornIn`, `plays`, `memberOf`, `signedWith`, `recordedOn`, `playedBy`, `releasedBy`, `hasGenre`, `basedIn`, and `influencedBy`. Domain and range restrictions are declared to enable downstream OWL reasoning.

### 3.2 KG Building

`src/kg/build_kg.py` reads the extracted CSV and produces RDF triples in Turtle format. URIs are generated deterministically from normalized entity strings to ensure stable, reproducible identifiers. Entity deduplication merges surface-form variants before triple emission, reducing noise in the resulting graph (`initial_kg.ttl`).

### 3.3 Wikidata Alignment

`src/kg/alignment.py` aligns local entities to Wikidata using the `wbsearchentities` REST API. For well-known jazz artists, hardcoded QIDs serve as ground-truth anchors, guaranteeing coverage for the most important nodes. Each successful match produces an `owl:sameAs` triple in `alignment.ttl`. Alignment coverage meets the ≥50% threshold required by the lab contract, enabling authoritative provenance links.

### 3.4 SPARQL Expansion

`src/kg/expand_kb.py` performs 1-hop and 2-hop SPARQL queries against the Wikidata endpoint, targeting jazz-relevant predicates: `P31` (instance of), `P175` (performer), `P264` (record label), `P1303` (instrument), `P136` (genre), `P19` (place of birth), `P569` (date of birth), and `P737` (influenced by). The result, `expanded.nt`, reaches approximately **50,000 triples**, enriching the KG with biographical, discographic, and stylistic facts beyond what Wikipedia text alone provides.

**Pipeline overview:**

```
Wikipedia seed URLs
  └─[crawler.py]──────────► crawler_output.jsonl    (≥20 pages, ≥500 words)
       └─[ner_pipeline.py]─► extracted_knowledge.csv (600+ entities)
            └─[ontology.py]─► ontology.ttl           (7 classes, 10+ properties)
                 └─[build_kg.py]──► initial_kg.ttl   (RDF triples)
                      └─[alignment.py]──► alignment.ttl  (owl:sameAs, ≥50% coverage)
                           └─[expand_kb.py]──► expanded.nt (~50k triples)
```

---

## 4. OWL Reasoning

A custom forward-chaining inference engine applies four rules at KG load time:

1. **Label inference**: if a musician `recordedOn` an album and that album `releasedBy` a label, infer `musician signedWith label`.
2. **Location inheritance**: band members inherit `bornIn` from the band's `basedIn` location when no individual birthplace is recorded.
3. **Influence transitivity**: two-hop `influencedBy` chains are materialized (A influenced B, B influenced C → A influenced C).
4. **RDFS subClassOf closure**: inferred types are propagated through the class hierarchy.

A consistency checker detects type contradictions (e.g., an entity simultaneously typed as both `Musician` and `RecordLabel`). This lightweight reasoner avoids the overhead of a full OWL-DL engine while providing practically useful inferred triples for downstream queries.

---

## 5. Knowledge Graph Embeddings

Two KGE models were trained on the expanded graph using a pure NumPy implementation (100 epochs, embedding dimension 50, learning rate 0.01, margin 1.0, batch size 256).

| Model    | MRR   | Hits@1 | Hits@3 | Hits@10 |
|----------|-------|--------|--------|---------|
| TransE   | 0.102 | 0.044  | 0.103  | 0.191   |
| DistMult | 0.076 | 0.042  | 0.079  | 0.144   |

TransE outperforms DistMult across all metrics, consistent with its known strength on sparse, low-degree graphs where translational geometry provides a stronger inductive bias than bilinear scoring. The absolute MRR values are modest for several reasons: (1) the graph is **sparse** — many entities appear only once or twice, providing insufficient training signal; (2) the NumPy implementation applies **no entity normalization** between steps, allowing embedding norms to drift; (3) 100 epochs at dimension 50 is insufficient for a heterogeneous graph of this complexity. A production library such as PyKEEN, which applies L2 normalization, optimized negative sampling, and early stopping, would likely double the MRR on the same graph. Despite low ranking metrics, the t-SNE and PCA projections reveal meaningful clustering by ontology class, confirming that the models capture structural regularity.

---

## 6. RAG Pipeline & NL→SPARQL Interface

The question-answering component converts natural language queries into SPARQL using Mistral 7B, served locally via the Ollama REST API. The system prompt provides a KG schema summary and six in-context SPARQL examples grounding the model in the graph's vocabulary and URI patterns.

A **self-repair loop** (up to 3 attempts) handles the most notable failure mode: if the generated SPARQL raises a parse error or returns no results, the error message is fed back to the model, allowing it to self-correct without user intervention.

A **template matcher** for 11 common question patterns (e.g., "Who plays trumpet?", "Which albums were released on Blue Note?") bypasses the LLM entirely, delivering fast, deterministic responses. Post-processing normalizes common LLM output artifacts: `/* */` block comments are stripped, missing prefix declarations are auto-injected, hallucinated functions like `STRCASEMATCH` are rewritten as `LCASE`, and spurious hardcoded `wd:Q...` literal filters are removed.

**Limitations**: Mistral 7B occasionally generates plausible but incorrect property URIs for questions not covered by the few-shot examples. Multi-hop queries require decomposition that the current single-pass prompt does not reliably handle.

---

## 7. Application & Visualization

The Streamlit application exposes three tabs:

- **RAG Demo**: accepts a natural language question, displays the generated SPARQL, result table, and repair attempt count.
- **Overview & Stats**: shows triple counts, entity type distribution, the KGE metrics table, and the t-SNE embedding visualization.
- **Graph Explorer**: renders an interactive PyVis network with node selection, type filtering, and edge traversal.

OWL reasoning is applied at application load time, so all inferred triples are immediately available for querying without a separate preprocessing step.

---

## 8. Reflection & Limitations

The pipeline is fully replicable: `run_pipeline.sh` executes all six steps in order, dependencies are pinned in `requirements.txt`, and 56 pytest tests validate every intermediate data contract. This reproducibility is a deliberate design priority.

Key limitations:

- **NER noise**: dependency-based relation extraction produces false positives in complex sentences. A supervised relation extraction model would improve precision.
- **Sparse alignment**: entities not covered by hardcoded QIDs rely on string-match API calls, which are sensitive to name variations.
- **KGE quality**: the NumPy implementation and sparse graph limit embedding performance. PyKEEN on a denser graph (~200k triples) would yield substantially better metrics.
- **LLM hallucination**: the NL→SPARQL interface degrades for uncommon predicates outside the few-shot examples.

---

## 9. Conclusion

This project demonstrates a complete, end-to-end knowledge graph pipeline for the Jazz domain — from raw web crawling to interactive querying and embedding-based analysis. The system successfully integrates classical semantic web infrastructure (OWL, SPARQL, `owl:sameAs`) with modern NLP components (transformer NER, LLM-driven query generation) and machine learning (KGE). The modular architecture, comprehensive test suite, and Streamlit interface collectively satisfy both the technical requirements of the lab contract and the broader goal of a replicable, extensible research prototype. Future work would focus on higher-quality relation extraction, scalable KGE training with PyKEEN, and multi-hop reasoning in the NL→SPARQL layer.
