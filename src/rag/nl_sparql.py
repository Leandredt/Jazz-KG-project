"""
NL→SPARQL Pipeline
==================
Converts natural-language questions into SPARQL queries using an Ollama LLM,
executes them against the Jazz KG, and self-repairs on failure (up to 3 tries).

Main interface:
    nl = NLToSPARQL(g, model="mistral:7b")
    result = nl.answer("Who are the musicians that play trumpet?")
"""

import logging
import re
from typing import Optional

import requests
from rdflib import Graph
from rdflib.exceptions import ParserError

from rag.schema_summary import build_schema_summary

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 120  # seconds

# Pre-built queries for common questions — bypass the LLM entirely.
# Keys are lowercase keyword tuples that must ALL appear in the question.
_SPARQL_PREFIX = """\
PREFIX jazz: <http://jazz-kg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

QUERY_TEMPLATES: list[tuple[tuple[str, ...], str]] = [
    (
        ("trumpet",),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?musicianLabel WHERE {
  ?musician a jazz:Musician ; jazz:plays ?inst ; rdfs:label ?musicianLabel .
  ?inst rdfs:label ?instLabel .
  FILTER(CONTAINS(LCASE(STR(?instLabel)), "trumpet"))
} LIMIT 20""",
    ),
    (
        ("instrument",),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?musicianLabel ?instLabel WHERE {
  ?musician a jazz:Musician ; jazz:plays ?inst ; rdfs:label ?musicianLabel .
  ?inst rdfs:label ?instLabel .
} LIMIT 20""",
    ),
    (
        ("miles davis", "album"),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?albumLabel WHERE {
  ?musician rdfs:label ?mLabel .
  FILTER(CONTAINS(LCASE(STR(?mLabel)), "miles davis"))
  ?album jazz:playedBy ?musician ; rdfs:label ?albumLabel .
} LIMIT 20""",
    ),
    (
        ("album",),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?albumLabel ?artistLabel WHERE {
  ?album a jazz:Album ; rdfs:label ?albumLabel .
  OPTIONAL { ?album jazz:playedBy ?m . ?m rdfs:label ?artistLabel . }
} LIMIT 20""",
    ),
    (
        ("new orleans",),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?musicianLabel WHERE {
  ?musician jazz:bornIn ?loc ; rdfs:label ?musicianLabel .
  ?loc rdfs:label ?locLabel .
  FILTER(CONTAINS(LCASE(STR(?locLabel)), "new orleans"))
} LIMIT 20""",
    ),
    (
        ("born",),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?musicianLabel ?locLabel WHERE {
  ?musician jazz:bornIn ?loc ; rdfs:label ?musicianLabel .
  ?loc rdfs:label ?locLabel .
} LIMIT 20""",
    ),
    (
        ("charlie parker", "genre"),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?genreLabel WHERE {
  ?musician rdfs:label ?mLabel ; jazz:hasGenre ?genre .
  FILTER(CONTAINS(LCASE(STR(?mLabel)), "charlie parker"))
  ?genre rdfs:label ?genreLabel .
} LIMIT 20""",
    ),
    (
        ("genre",),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?musicianLabel ?genreLabel WHERE {
  ?musician jazz:hasGenre ?genre ; rdfs:label ?musicianLabel .
  ?genre rdfs:label ?genreLabel .
} LIMIT 20""",
    ),
    (
        ("bebop", "label"),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?labelLabel WHERE {
  ?musician jazz:hasGenre ?genre ; rdfs:label ?mLabel .
  ?genre rdfs:label ?genreLabel .
  FILTER(CONTAINS(LCASE(STR(?genreLabel)), "bebop"))
  ?album jazz:playedBy ?musician ; jazz:releasedBy ?label .
  ?label rdfs:label ?labelLabel .
} LIMIT 20""",
    ),
    (
        ("record label", "album"),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?labelLabel ?albumLabel WHERE {
  ?album jazz:releasedBy ?label ; rdfs:label ?albumLabel .
  ?label rdfs:label ?labelLabel .
} LIMIT 20""",
    ),
    (
        ("blue note",),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?albumLabel WHERE {
  ?album jazz:releasedBy ?label ; rdfs:label ?albumLabel .
  ?label rdfs:label ?labelLabel .
  FILTER(CONTAINS(LCASE(STR(?labelLabel)), "blue note"))
} LIMIT 20""",
    ),
    (
        ("jazz fusion", "instrument"),
        _SPARQL_PREFIX + """\
SELECT DISTINCT ?instLabel WHERE {
  ?musician jazz:hasGenre ?genre ; jazz:plays ?inst ; rdfs:label ?mLabel .
  ?genre rdfs:label ?genreLabel .
  FILTER(CONTAINS(LCASE(STR(?genreLabel)), "jazz fusion"))
  ?inst rdfs:label ?instLabel .
} LIMIT 20""",
    ),
]


def _match_template(question: str) -> str | None:
    """Return a pre-built SPARQL query if the question matches a template, else None."""
    q = question.lower()
    for keywords, sparql in QUERY_TEMPLATES:
        if all(kw in q for kw in keywords):
            return sparql
    return None


SUGGESTED_QUESTIONS = [
    "Who are the musicians that play trumpet?",
    "What albums did Miles Davis record?",
    "Which musicians were born in New Orleans?",
    "Who influenced John Coltrane?",
    "What genres did Charlie Parker play?",
    "Which record labels released bebop albums?",
    "What instruments are associated with jazz fusion?",
]

PROMPT_TEMPLATE = """\
You are a SPARQL query generator for a Jazz Knowledge Graph. Your only job is to output a single valid SPARQL SELECT query.

SCHEMA:
{schema}

QUESTION: {question}

STRICT RULES — follow every rule exactly:

1. PREFIXES — always declare exactly these, no others:
   PREFIX jazz: <http://jazz-kg.org/ontology#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX wdt: <http://www.wikidata.org/prop/direct/>
   PREFIX wd: <http://www.wikidata.org/entity/>

2. ALLOWED keywords: SELECT DISTINCT, WHERE, OPTIONAL, FILTER, BIND, LIMIT.
   FORBIDDEN keywords: IF, EXISTS, NOT EXISTS, MINUS, subqueries, VALUES, HAVING.

3. ALLOWED functions: LCASE, STR, CONTAINS, STRAFTER, COALESCE, COUNT, GROUP BY.
   FORBIDDEN functions: LANG, DATATYPE, LANGMATCHES, REGEX, STRSTARTS, STRENDS, and any function not listed above.

4. STRING MATCHING — always use this exact pattern:
   FILTER(CONTAINS(LCASE(STR(?var)), "keyword"))

5. LABELS — every URI must be joined to its rdfs:label to get a human-readable value.
   Pattern: ?uri rdfs:label ?uriLabel .

6. OUTPUT — LIMIT 20. Return only SELECT queries, never ASK/CONSTRUCT/DESCRIBE.

7. PREDICATES — always prefer jazz: predicates over wdt: predicates.
   Use wdt: ONLY when the question explicitly asks about Wikidata properties.
   NEVER hardcode wd:Q... URIs in FILTER clauses — always match via rdfs:label.
   FORBIDDEN: comments in any form (no #, no //, no /* */).

EXAMPLES:

Example 1 — musicians playing an instrument:
```sparql
PREFIX jazz: <http://jazz-kg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?musicianLabel ?instrumentLabel WHERE {{
  ?musician a jazz:Musician ;
            jazz:plays ?instrument ;
            rdfs:label ?musicianLabel .
  ?instrument rdfs:label ?instrumentLabel .
  FILTER(CONTAINS(LCASE(STR(?instrumentLabel)), "trumpet"))
}} LIMIT 20
```

Example 2 — albums by an artist:
```sparql
PREFIX jazz: <http://jazz-kg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?albumLabel WHERE {{
  ?musician rdfs:label ?musicianLabel .
  FILTER(CONTAINS(LCASE(STR(?musicianLabel)), "miles davis"))
  ?album jazz:playedBy ?musician ;
         rdfs:label ?albumLabel .
}} LIMIT 20
```

Example 3 — genres of an artist:
```sparql
PREFIX jazz: <http://jazz-kg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?genreLabel WHERE {{
  ?musician rdfs:label ?musicianLabel ;
            jazz:hasGenre ?genre .
  FILTER(CONTAINS(LCASE(STR(?musicianLabel)), "charlie parker"))
  ?genre rdfs:label ?genreLabel .
}} LIMIT 20
```

Example 4 — albums released by a record label:
```sparql
PREFIX jazz: <http://jazz-kg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?albumLabel WHERE {{
  ?album jazz:releasedBy ?label ;
         rdfs:label ?albumLabel .
  ?label rdfs:label ?labelName .
  FILTER(CONTAINS(LCASE(STR(?labelName)), "blue note"))
}} LIMIT 20
```

Example 5 — record labels that released albums in a genre (jazz:hasGenre is on musicians, not albums):
```sparql
PREFIX jazz: <http://jazz-kg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?labelLabel WHERE {{
  ?musician jazz:hasGenre ?genre ;
            rdfs:label ?musicianLabel .
  ?genre rdfs:label ?genreLabel .
  FILTER(CONTAINS(LCASE(STR(?genreLabel)), "bebop"))
  ?album jazz:playedBy ?musician ;
         jazz:releasedBy ?label .
  ?label rdfs:label ?labelLabel .
}} LIMIT 20
```

Example 6 — musicians born in a city:
```sparql
PREFIX jazz: <http://jazz-kg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?musicianLabel WHERE {{
  ?musician a jazz:Musician ;
            jazz:bornIn ?location ;
            rdfs:label ?musicianLabel .
  ?location rdfs:label ?locLabel .
  FILTER(CONTAINS(LCASE(STR(?locLabel)), "new orleans"))
}} LIMIT 20
```

Return ONLY the SPARQL query inside ```sparql ... ``` markers. No explanation.
"""

REPAIR_TEMPLATE = """\
This SPARQL query failed. Rewrite it from scratch to fix the error.

FAILED QUERY:
```sparql
{sparql}
```

ERROR: {error}

STRICT RULES (same as before):
1. Allowed prefixes: jazz: rdfs: wdt: wd: — no others.
2. Allowed keywords: SELECT DISTINCT WHERE OPTIONAL FILTER BIND LIMIT — no IF, EXISTS, subqueries.
3. Allowed functions: LCASE STR CONTAINS STRAFTER COALESCE COUNT — no LANG, REGEX, STRSTARTS, or any other.
4. String matching: FILTER(CONTAINS(LCASE(STR(?var)), "keyword")) — always wrap with STR().
5. Every URI variable must be joined to rdfs:label for display.
6. LIMIT 20.

SCHEMA:
{schema}

Return ONLY the corrected SPARQL query inside ```sparql ... ``` markers.
"""


class NLToSPARQL:
    """
    Natural-language to SPARQL pipeline backed by an Ollama LLM.

    Parameters
    ----------
    g : rdflib.Graph
        Loaded RDF graph to query.
    model : str
        Ollama model name (default: mistral:7b).
    """

    def __init__(self, g: Graph, model: str = "mistral:7b"):
        self.g = g
        self.model = model
        self._schema: Optional[str] = None  # built lazily
        logger.info("NLToSPARQL initialised with model=%s, %d triples", model, len(g))

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @property
    def schema(self) -> str:
        if self._schema is None:
            self._schema = build_schema_summary(self.g)
        return self._schema

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, question: str, schema: str) -> str:
        """Build the full prompt from the template."""
        return PROMPT_TEMPLATE.format(schema=schema, question=question)

    # ------------------------------------------------------------------
    # Ollama call
    # ------------------------------------------------------------------

    def query_ollama(self, prompt: str) -> str:
        """
        POST to Ollama REST API and return the model's response text.
        Raises RuntimeError if Ollama is unreachable.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Ollama is not running at {OLLAMA_URL}. "
                "Start it with: ollama serve"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama timed out after {OLLAMA_TIMEOUT}s"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # SPARQL extraction
    # ------------------------------------------------------------------

    def extract_sparql(self, response: str) -> str:
        """
        Extract the SPARQL block from a markdown-fenced LLM response.
        Falls back to the full response if no fences are found.
        """
        # Match ```sparql ... ``` or ``` ... ```
        pattern = r"```(?:sparql)?\s*(.*?)```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        sparql = match.group(1).strip() if match else response.strip()
        # Clean up common LLM mistakes
        sparql = re.sub(r"(?<!:)//[^\n]*", "", sparql)   # remove // comments but not http:// URIs
        sparql = re.sub(r"/\*.*?\*/", "", sparql, flags=re.DOTALL)  # remove /* */ block comments
        sparql = re.sub(r"\bwd:\"(Q\d+)\"", r"wd:\1", sparql)  # fix wd:"Q123" → wd:Q123
        sparql = re.sub(r"\n{3,}", "\n\n", sparql)       # collapse blank lines
        # Fix invalid/hallucinated SPARQL functions
        sparql = re.sub(r"\bSTRCASEMATIC\b", "LCASE", sparql, flags=re.IGNORECASE)
        sparql = re.sub(r"\bSTRCASE\b", "LCASE", sparql, flags=re.IGNORECASE)
        # Fix strict LANG filters that exclude plain literals (no language tag)
        sparql = re.sub(
            r'FILTER\s*\(\s*LANG\s*\((\s*\?[\w]+\s*)\)\s*=\s*"en"\s*\)',
            r'FILTER(LANG(\1) = "en" || LANG(\1) = "")',
            sparql,
        )
        # Remove hardcoded wd:Q... URI comparisons — always wrong, causes parse errors
        sparql = re.sub(r'FILTER\s*\([\s\S]*?wd:Q\d+[\s\S]*?\)\s*\.?', '', sparql)
        # Inject missing standard prefixes that are referenced but not declared
        _STANDARD_PREFIXES = [
            (r'\brdfs:', "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"),
            (r'\brdf:(?!s)', "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"),
            (r'\bowl:', "PREFIX owl: <http://www.w3.org/2002/07/owl#>"),
            (r'\bxsd:', "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"),
        ]
        for pattern, declaration in _STANDARD_PREFIXES:
            if re.search(pattern, sparql) and declaration not in sparql:
                sparql = declaration + "\n" + sparql
        return sparql.strip()

    # ------------------------------------------------------------------
    # SPARQL execution
    # ------------------------------------------------------------------

    def execute_sparql(self, sparql: str) -> list:
        """
        Run a SPARQL SELECT query on the internal graph.
        Returns a list of result rows (each row is a dict of variable→value).
        Raises ValueError on parse/execution errors.
        """
        try:
            results = self.g.query(sparql)
            rows = []
            for row in results:
                row_dict = {}
                for var in results.vars:
                    val = row[var]
                    row_dict[str(var)] = str(val) if val is not None else None
                rows.append(row_dict)
            return rows
        except Exception as exc:
            raise ValueError(f"SPARQL execution error: {exc}") from exc

    # ------------------------------------------------------------------
    # Self-repair
    # ------------------------------------------------------------------

    def self_repair(self, sparql: str, error: str, attempt: int = 1) -> str:
        """
        Ask the LLM to fix a broken SPARQL query.
        Returns a new SPARQL string (not yet executed).
        """
        logger.info("Self-repair attempt %d for error: %s", attempt, error[:120])
        repair_prompt = REPAIR_TEMPLATE.format(
            sparql=sparql,
            error=error,
            schema=self.schema,
        )
        raw_response = self.query_ollama(repair_prompt)
        return self.extract_sparql(raw_response)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def answer(self, question: str) -> dict:
        """
        Full NL→SPARQL pipeline for one question.

        Returns a dict with keys:
          question, sparql, results, success, repairs_needed,
          error (if any), answer_text
        """
        result = {
            "question": question,
            "sparql": None,
            "results": [],
            "success": False,
            "repairs_needed": 0,
            "error": None,
            "answer_text": "",
        }

        # --- Step 0: check template cache (bypass LLM for known question patterns) ---
        template_sparql = _match_template(question)
        if template_sparql is not None:
            logger.info("Template match for question: %s", question)
            result["sparql"] = template_sparql
            try:
                result["results"] = self.execute_sparql(template_sparql)
                result["success"] = True
            except ValueError as exc:
                result["error"] = str(exc)
            result["answer_text"] = self._format_answer(question, result)
            return result

        # --- Step 1: build prompt and call LLM ---
        prompt = self.build_prompt(question, self.schema)
        try:
            raw_response = self.query_ollama(prompt)
        except RuntimeError as exc:
            result["error"] = str(exc)
            result["answer_text"] = f"[LLM unavailable] {exc}"
            return result

        sparql = self.extract_sparql(raw_response)
        result["sparql"] = sparql

        # --- Step 2: execute with self-repair loop ---
        MAX_REPAIRS = 3
        for attempt in range(MAX_REPAIRS + 1):
            try:
                rows = self.execute_sparql(sparql)
                result["results"] = rows
                result["success"] = True
                break
            except ValueError as exc:
                error_msg = str(exc)
                result["error"] = error_msg
                if attempt >= MAX_REPAIRS:
                    logger.warning("All %d repair attempts exhausted.", MAX_REPAIRS)
                    break
                try:
                    sparql = self.self_repair(sparql, error_msg, attempt + 1)
                    result["sparql"] = sparql
                    result["repairs_needed"] += 1
                except RuntimeError as repair_exc:
                    result["error"] = str(repair_exc)
                    break

        # --- Step 3: format answer text ---
        result["answer_text"] = self._format_answer(question, result)
        return result

    # ------------------------------------------------------------------
    # Answer formatting
    # ------------------------------------------------------------------

    def _format_answer(self, question: str, result: dict) -> str:
        if not result["success"]:
            return (
                f"Could not generate a valid SPARQL answer for: {question}\n"
                f"Error: {result.get('error', 'unknown')}"
            )
        rows = result["results"]
        if not rows:
            return f"No results found in the KG for: {question}"

        lines = [f"Answer for: {question}", f"({len(rows)} result(s) found)\n"]
        for i, row in enumerate(rows[:20], 1):
            values = [v for v in row.values() if v is not None]
            lines.append(f"  {i}. {' | '.join(values)}")
        if len(rows) > 20:
            lines.append(f"  ... and {len(rows) - 20} more.")
        return "\n".join(lines)
