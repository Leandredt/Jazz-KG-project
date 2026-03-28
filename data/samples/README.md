# Lab 1 Data Samples — Format Reference

This directory may contain sample files that illustrate the expected format of
Lab 1 outputs. The canonical outputs live in `data/`.

---

## 1. `data/crawler_output.jsonl` — Crawler Output

**Format**: JSONL (JSON Lines) — one JSON object per line, UTF-8 encoded.

### Schema

| Field        | Type    | Description |
|--------------|---------|-------------|
| `url`        | string  | Canonical Wikipedia article URL (`https://en.wikipedia.org/wiki/...`) |
| `title`      | string  | Article title (without " - Wikipedia" suffix) |
| `text`       | string  | Clean article body text extracted by trafilatura (no HTML, no templates) |
| `word_count` | integer | Number of whitespace-separated tokens in `text` (always >= 500) |

### Example record

```json
{
  "url": "https://en.wikipedia.org/wiki/Miles_Davis",
  "title": "Miles Davis",
  "text": "Miles Dewey Davis III (May 26, 1926 – September 28, 1991) was an American jazz trumpeter, bandleader, and composer. He is among the most influential and acclaimed figures in the history of jazz and 20th century music...",
  "word_count": 8234
}
```

### Constraints

- No duplicate URLs
- `word_count` >= 500 (short pages are discarded by the crawler)
- All URLs match the pattern `https://en.wikipedia.org/wiki/*`
- No Wikipedia special-namespace pages (Category:, Talk:, etc.)

---

## 2. `data/extracted_knowledge.csv` — NER + Relation Extraction Output

**Format**: CSV with header row, UTF-8 encoded, comma-separated.

### Columns

| Column             | Description |
|--------------------|-------------|
| `entity`           | Named entity string as extracted by spaCy (e.g. `Miles Davis`) |
| `entity_type`      | spaCy NER label — one of `PERSON`, `ORG`, `GPE`, `DATE`, `WORK_OF_ART` |
| `source_url`       | Wikipedia URL the entity was found on (matches a URL in `crawler_output.jsonl`) |
| `context_sentence` | The full sentence in which the entity appeared |
| `relation_to`      | Target entity of a dependency-parsed relation (empty string if none found) |
| `relation_type`    | Lemma of the verb or preposition linking `entity` to `relation_to` (empty if none) |

### Entity types and their jazz-domain meaning

| `entity_type`  | Jazz-domain interpretation | Examples |
|----------------|---------------------------|----------|
| `PERSON`       | Jazz musician or vocalist  | `Miles Davis`, `John Coltrane`, `Billie Holiday` |
| `ORG`          | Record label or band/group | `Blue Note Records`, `Miles Davis Quintet` |
| `GPE`          | City, state, or country    | `New Orleans`, `New York`, `Chicago` |
| `DATE`         | Year or time period        | `1959`, `the 1940s`, `September 28, 1991` |
| `WORK_OF_ART`  | Album or composition title | `Kind of Blue`, `A Love Supreme` |

### Example rows

```
entity,entity_type,source_url,context_sentence,relation_to,relation_type
Miles Davis,PERSON,https://en.wikipedia.org/wiki/Miles_Davis,"Miles Davis was an American jazz trumpeter, bandleader, and composer.",,
Kind of Blue,WORK_OF_ART,https://en.wikipedia.org/wiki/Kind_of_Blue,"Kind of Blue is a studio album by American jazz musician Miles Davis.",Miles Davis,release
Blue Note Records,ORG,https://en.wikipedia.org/wiki/John_Coltrane,"Coltrane signed with Blue Note Records in 1957.",Coltrane,sign
New Orleans,GPE,https://en.wikipedia.org/wiki/Louis_Armstrong,"Louis Armstrong was born in New Orleans, Louisiana.",,
1959,DATE,https://en.wikipedia.org/wiki/Kind_of_Blue,"Kind of Blue was recorded on March 2 and April 22, 1959.",,
```

### Notes for Lab 2

- Rows where `relation_to` is non-empty define candidate **knowledge graph edges**:
  `entity --[relation_type]--> relation_to`
- Rows where `relation_to` is empty represent **isolated entities** (nodes without
  a known edge from this sentence — edges may come from other rows)
- Deduplicate on `(entity, entity_type, relation_to, source_url)` before building
  the graph to avoid spurious duplicate edges
- Instrument names (saxophone, trumpet, piano) are common nouns and are **not**
  extracted as named entities; use dependency parsing on `context_sentence` or
  Wikidata for instrument enrichment

---

## 3. `data/lab1_contract.json` — Interface Contract

Machine-readable description of all Lab 1 outputs, consumed by Lab 2 to
validate inputs before starting graph construction. See that file for the
full schema.
