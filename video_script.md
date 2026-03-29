# Jazz Knowledge Graph — Video Script (~5 minutes)

---

**[SCREEN: README or project folder structure]**

Hi, in this video I'll walk you through the Jazz Knowledge Graph project — a full end-to-end pipeline that goes from raw Wikipedia text to a queryable, interactive knowledge base with a natural language interface and machine learning embeddings.

Jazz music is a rich domain with well-documented artists, albums, labels, and influences — which makes it a great candidate for a knowledge graph. The goal was to automatically extract that knowledge from the web, structure it semantically, enrich it with Wikidata, and make it accessible through a modern interface.

---

**[SCREEN: `crawler_output.jsonl` in terminal — show one JSON entry]**

Everything starts with a Wikipedia crawler. It targets pages on jazz artists, albums, bands, and record labels. We use `httpx` for requests and `trafilatura` to strip boilerplate and keep only the article text. The crawler respects robots.txt, waits 1.5 seconds between requests, and collects up to 120 pages — keeping only those with at least 500 words. We end up with well over 20 high-quality documents.

---

**[SCREEN: `extracted_knowledge.csv` — show a few rows]**

The raw text then goes through a Named Entity Recognition pipeline using spaCy's transformer model `en_core_web_trf`. It identifies persons, organizations, locations, dates, and works of art. Beyond detection, we also use dependency parsing to extract relations between entities — for example, a musician playing an instrument, or an artist belonging to a band.

The result is a CSV with over 600 entity records, each with its type, source URL, a context sentence, and any relation it holds to another entity.

---

**[SCREEN: `ontology.ttl` briefly, then terminal showing triple count]**

From the CSV, we build an OWL ontology with seven classes — Musician, Album, RecordLabel, Band, Location, Instrument, Genre — and over ten object properties like `bornIn`, `plays`, `releasedBy`, and `influencedBy`. Entities from the CSV are mapped into RDF triples using deterministic URI generation to avoid duplicates.

We then align our entities to Wikidata using its API, producing `owl:sameAs` links for over 50% of the graph. Finally, a SPARQL expansion queries Wikidata for additional biographical and discographic facts in a 1 and 2-hop traversal — reaching approximately 50,000 triples in the final graph.

---

**[SCREEN: Streamlit app — RAG Demo tab]**

Now the application. The main feature is the RAG Demo tab. The user types a natural language question, and Mistral 7B — running locally via Ollama — translates it into SPARQL. The model receives the question along with the KG schema and example queries as context. If the query fails, a self-repair loop feeds the error back to the model and retries up to three times. For common question patterns, a template matcher bypasses the LLM entirely for faster, more reliable results.

---

**[SCREEN: Type "What albums did Miles Davis record?" — show SPARQL + results]**

Let's ask: *"What albums did Miles Davis record?"* — Kind of Blue, Miles Smiles, Bitches Brew. The SPARQL is generated and executed in under a second.

---

**[SCREEN: Type "What genres did Charlie Parker play?" — show results]**

And: *"What genres did Charlie Parker play?"* — this one hits the template matcher directly. Answer: Bebop.

---

**[SCREEN: Overview & Stats tab — KGE table + t-SNE image]**

In the Overview tab, our embedding results. We trained TransE and DistMult for 100 epochs at dimension 50. TransE achieves MRR 0.102 and Hits@10 of 19%. The scores are modest — the graph is sparse and the NumPy implementation lacks the normalization that a library like PyKEEN would apply. But the t-SNE projection shows that embeddings do learn structure: musicians, albums, and labels cluster into distinct regions.

---

**[SCREEN: Graph Explorer tab]**

The Graph Explorer renders an interactive network — you can select a node, expand its neighborhood, and navigate entity relationships visually.

---

**[SCREEN: RAG Demo or README]**

To wrap up: this project covers the full KG lifecycle — crawling, extraction, ontology design, alignment, expansion, reasoning, embeddings, and a natural language interface. The pipeline is fully replicable with a single shell script and validated by 56 automated tests. Thanks for watching.
