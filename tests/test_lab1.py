"""
Lab 1 Tests - Jazz Knowledge Graph
====================================
Validates that the crawler and NER pipeline produced correct output files
meeting the minimum quality thresholds defined in lab1_contract.json.

Run with:
    python -m pytest tests/test_lab1.py -v

Or directly:
    python tests/test_lab1.py
"""

import csv
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CRAWLER_OUTPUT = PROJECT_ROOT / "data" / "crawler_output.jsonl"
NER_OUTPUT = PROJECT_ROOT / "data" / "extracted_knowledge.csv"
CONTRACT_FILE = PROJECT_ROOT / "data" / "lab1_contract.json"

# ---------------------------------------------------------------------------
# Thresholds (should match lab1_contract.json)
# ---------------------------------------------------------------------------

MIN_CRAWLER_PAGES = 20
MIN_NER_ENTITIES = 100
MIN_WORD_COUNT_PER_PAGE = 500
REQUIRED_ENTITY_TYPES = {"PERSON", "ORG", "GPE"}
REQUIRED_CSV_COLUMNS = {
    "entity",
    "entity_type",
    "source_url",
    "context_sentence",
    "relation_to",
    "relation_type",
}


# ===========================================================================
# Crawler tests
# ===========================================================================

class TestCrawlerOutput:

    def test_crawler_output_file_exists(self):
        """crawler_output.jsonl must exist."""
        assert CRAWLER_OUTPUT.exists(), (
            f"Crawler output not found at {CRAWLER_OUTPUT}. "
            "Run: python src/crawl/crawler.py"
        )

    def test_crawler_output_not_empty(self):
        """File must not be empty."""
        assert CRAWLER_OUTPUT.stat().st_size > 0, "crawler_output.jsonl is empty"

    def test_crawler_output_valid_jsonl(self):
        """Every non-blank line must be valid JSON."""
        invalid_lines = []
        with open(CRAWLER_OUTPUT, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    invalid_lines.append((lineno, str(exc)))
        assert not invalid_lines, (
            f"Invalid JSONL on lines: {invalid_lines[:5]}"
        )

    def test_crawler_minimum_page_count(self):
        """Must have at least MIN_CRAWLER_PAGES entries."""
        count = _count_jsonl_lines(CRAWLER_OUTPUT)
        assert count >= MIN_CRAWLER_PAGES, (
            f"Expected >= {MIN_CRAWLER_PAGES} pages, got {count}"
        )

    def test_crawler_required_fields(self):
        """Every record must have url, title, text, word_count fields."""
        required = {"url", "title", "text", "word_count"}
        records = _load_jsonl(CRAWLER_OUTPUT)
        for i, record in enumerate(records):
            missing = required - set(record.keys())
            assert not missing, (
                f"Record {i} missing fields: {missing}. Record keys: {set(record.keys())}"
            )

    def test_crawler_word_count_above_minimum(self):
        """Every stored page must have word_count >= MIN_WORD_COUNT_PER_PAGE."""
        records = _load_jsonl(CRAWLER_OUTPUT)
        short_pages = [
            (r.get("url", "?"), r.get("word_count", 0))
            for r in records
            if r.get("word_count", 0) < MIN_WORD_COUNT_PER_PAGE
        ]
        assert not short_pages, (
            f"{len(short_pages)} pages below {MIN_WORD_COUNT_PER_PAGE} words: "
            f"{short_pages[:5]}"
        )

    def test_crawler_word_count_matches_text(self):
        """word_count field should roughly match actual text word count."""
        records = _load_jsonl(CRAWLER_OUTPUT)
        for record in records[:10]:  # sample first 10
            declared = record.get("word_count", 0)
            actual = len(record.get("text", "").split())
            # Allow 10% tolerance
            assert abs(declared - actual) <= max(10, actual * 0.10), (
                f"word_count mismatch for {record.get('url')}: "
                f"declared={declared}, actual={actual}"
            )

    def test_crawler_urls_are_wikipedia(self):
        """All stored URLs must be Wikipedia article URLs."""
        records = _load_jsonl(CRAWLER_OUTPUT)
        non_wiki = [
            r.get("url", "")
            for r in records
            if "wikipedia.org/wiki/" not in r.get("url", "")
        ]
        assert not non_wiki, (
            f"Non-Wikipedia URLs found: {non_wiki[:5]}"
        )

    def test_crawler_no_duplicate_urls(self):
        """No two records should have the same URL."""
        records = _load_jsonl(CRAWLER_OUTPUT)
        urls = [r.get("url", "") for r in records]
        duplicates = [u for u in urls if urls.count(u) > 1]
        unique_dupes = list(dict.fromkeys(duplicates))
        assert not unique_dupes, (
            f"Duplicate URLs found: {unique_dupes[:5]}"
        )


# ===========================================================================
# NER / extracted knowledge tests
# ===========================================================================

class TestExtractedKnowledge:

    def test_ner_output_file_exists(self):
        """extracted_knowledge.csv must exist."""
        assert NER_OUTPUT.exists(), (
            f"NER output not found at {NER_OUTPUT}. "
            "Run: python src/ie/ner_pipeline.py"
        )

    def test_ner_output_not_empty(self):
        """File must not be empty."""
        assert NER_OUTPUT.stat().st_size > 0, "extracted_knowledge.csv is empty"

    def test_ner_csv_has_correct_columns(self):
        """CSV header must contain all required columns."""
        with open(NER_OUTPUT, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            actual_columns = set(reader.fieldnames or [])
        missing = REQUIRED_CSV_COLUMNS - actual_columns
        assert not missing, (
            f"CSV missing columns: {missing}. Found: {actual_columns}"
        )

    def test_ner_minimum_entity_count(self):
        """Must have at least MIN_NER_ENTITIES rows (excluding header)."""
        count = _count_csv_rows(NER_OUTPUT)
        assert count >= MIN_NER_ENTITIES, (
            f"Expected >= {MIN_NER_ENTITIES} entity rows, got {count}"
        )

    def test_ner_has_required_entity_types(self):
        """Must include at least PERSON, ORG, and GPE entities."""
        rows = _load_csv(NER_OUTPUT)
        found_types = {row["entity_type"] for row in rows}
        missing = REQUIRED_ENTITY_TYPES - found_types
        assert not missing, (
            f"Missing entity types: {missing}. Found: {found_types}"
        )

    def test_ner_entity_field_not_empty(self):
        """entity column must never be blank."""
        rows = _load_csv(NER_OUTPUT)
        blank = [i for i, r in enumerate(rows, start=2) if not r.get("entity", "").strip()]
        assert not blank, f"Blank entity values on CSV rows: {blank[:10]}"

    def test_ner_entity_type_valid_values(self):
        """entity_type must be one of the expected NLP types."""
        valid_types = {"PERSON", "ORG", "GPE", "DATE", "WORK_OF_ART",
                       "NORP", "FAC", "LOC", "PRODUCT", "EVENT", "LAW",
                       "LANGUAGE", "MONEY", "PERCENT", "QUANTITY",
                       "ORDINAL", "CARDINAL", "TIME"}
        rows = _load_csv(NER_OUTPUT)
        invalid = [
            (i, r["entity_type"])
            for i, r in enumerate(rows, start=2)
            if r.get("entity_type", "") not in valid_types
        ]
        assert not invalid, f"Invalid entity_type values: {invalid[:10]}"

    def test_ner_source_url_not_empty(self):
        """source_url column must never be blank."""
        rows = _load_csv(NER_OUTPUT)
        blank = [i for i, r in enumerate(rows, start=2) if not r.get("source_url", "").strip()]
        assert not blank, f"Blank source_url values on CSV rows: {blank[:10]}"

    def test_ner_context_sentence_not_empty(self):
        """context_sentence must never be blank."""
        rows = _load_csv(NER_OUTPUT)
        blank = [
            i for i, r in enumerate(rows, start=2)
            if not r.get("context_sentence", "").strip()
        ]
        assert not blank, f"Blank context_sentence values: {blank[:10]}"

    def test_ner_has_person_entities(self):
        """Must contain musician names (PERSON entities)."""
        rows = _load_csv(NER_OUTPUT)
        persons = [r["entity"] for r in rows if r.get("entity_type") == "PERSON"]
        assert len(persons) >= 10, (
            f"Expected >= 10 PERSON entities, got {len(persons)}"
        )

    def test_ner_has_org_entities(self):
        """Must contain label/band names (ORG entities)."""
        rows = _load_csv(NER_OUTPUT)
        orgs = [r["entity"] for r in rows if r.get("entity_type") == "ORG"]
        assert len(orgs) >= 5, (
            f"Expected >= 5 ORG entities, got {len(orgs)}"
        )

    def test_ner_has_gpe_entities(self):
        """Must contain city/country names (GPE entities)."""
        rows = _load_csv(NER_OUTPUT)
        gpes = [r["entity"] for r in rows if r.get("entity_type") == "GPE"]
        assert len(gpes) >= 5, (
            f"Expected >= 5 GPE entities, got {len(gpes)}"
        )

    def test_ner_has_relation_rows(self):
        """At least some rows must have a relation_to value (triplets)."""
        rows = _load_csv(NER_OUTPUT)
        with_relation = [r for r in rows if r.get("relation_to", "").strip()]
        assert len(with_relation) >= 10, (
            f"Expected >= 10 rows with relations, got {len(with_relation)}"
        )


# ===========================================================================
# Contract file tests
# ===========================================================================

class TestContractFile:

    def test_contract_file_exists(self):
        """lab1_contract.json must exist."""
        assert CONTRACT_FILE.exists(), f"Contract file not found: {CONTRACT_FILE}"

    def test_contract_file_valid_json(self):
        """lab1_contract.json must be valid JSON."""
        with open(CONTRACT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "Contract must be a JSON object"

    def test_contract_has_required_keys(self):
        """Contract must have outputs, entity_types, csv_columns keys."""
        with open(CONTRACT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("outputs", "entity_types", "csv_columns", "expected_entities_min"):
            assert key in data, f"Contract missing key: {key}"

    def test_contract_entity_types(self):
        """Contract entity_types must include the five required types."""
        with open(CONTRACT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        required = {"PERSON", "ORG", "GPE", "DATE", "WORK_OF_ART"}
        contract_types = set(data.get("entity_types", []))
        missing = required - contract_types
        assert not missing, f"Contract entity_types missing: {missing}"

    def test_contract_csv_columns(self):
        """Contract csv_columns must include all six required columns."""
        with open(CONTRACT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        contract_cols = set(data.get("csv_columns", []))
        missing = REQUIRED_CSV_COLUMNS - contract_cols
        assert not missing, f"Contract csv_columns missing: {missing}"


# ===========================================================================
# Cross-file consistency tests
# ===========================================================================

class TestCrossFileConsistency:

    def test_ner_source_urls_exist_in_crawler(self):
        """All source_url values in the CSV must appear in crawler_output.jsonl."""
        if not CRAWLER_OUTPUT.exists() or not NER_OUTPUT.exists():
            pytest.skip("Required files not present")
        crawler_urls = {r.get("url", "") for r in _load_jsonl(CRAWLER_OUTPUT)}
        ner_rows = _load_csv(NER_OUTPUT)
        ner_urls = {r.get("source_url", "") for r in ner_rows}
        phantom = ner_urls - crawler_urls
        assert not phantom, (
            f"NER references URLs not in crawler output: {list(phantom)[:5]}"
        )


# ===========================================================================
# Helpers
# ===========================================================================

def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _count_csv_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def _load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ===========================================================================
# Run as standalone script
# ===========================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
