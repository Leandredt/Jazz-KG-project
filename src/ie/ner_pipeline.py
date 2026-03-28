"""
Lab 1 - Jazz Knowledge Graph: NER & Relation Extraction Pipeline
=================================================================
Reads the JSONL output produced by the crawler, runs spaCy NER with the
transformer model (en_core_web_trf), extracts named entities and candidate
relation triplets via dependency parsing, and writes a CSV for Lab 2.

Usage:
    python src/ie/ner_pipeline.py

Input:
    data/crawler_output.jsonl

Output:
    data/extracted_knowledge.csv
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional

import spacy
from spacy.tokens import Doc, Span, Token

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "crawler_output.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "extracted_knowledge.csv"

SPACY_MODEL = "en_core_web_lg"

# Entity types we care about in the jazz domain
TARGET_ENT_TYPES = {"PERSON", "ORG", "GPE", "DATE", "WORK_OF_ART"}

# Dependency relation labels used for relation extraction
SUBJECT_DEPS = {"nsubj", "nsubjpass"}
OBJECT_DEPS = {"dobj", "pobj", "attr", "nmod", "appos"}

# Minimum character length for an entity string (avoids single letters etc.)
MIN_ENTITY_LENGTH = 2

# Maximum sentence length in tokens (very long sentences are slow + noisy)
MAX_SENTENCE_TOKENS = 100

# CSV column order (contract with Lab 2)
CSV_COLUMNS = [
    "entity",
    "entity_type",
    "source_url",
    "context_sentence",
    "relation_to",
    "relation_type",
]

# Regex patterns to remove Wikipedia annotation noise
_BRACKET_PATTERN = re.compile(r"\[\d+\]")   # citation markers like [1]
_EXTRA_SPACE = re.compile(r"\s{2,}")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jazz_ner")


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove citation noise and normalise whitespace."""
    text = _BRACKET_PATTERN.sub("", text)
    text = _EXTRA_SPACE.sub(" ", text)
    return text.strip()


def split_into_chunks(text: str, max_chars: int = 100_000) -> list[str]:
    """
    Split very long texts into paragraph-level chunks to keep spaCy memory
    usage bounded while preserving sentence boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append("\n".join(current))
    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Relation extraction helpers
# ---------------------------------------------------------------------------

def token_root(span: Span) -> Token:
    """Return the syntactic root token of an entity span."""
    for token in span:
        if token.head == token or token.dep_ == "ROOT":
            return token
        if token.head not in span:
            return token
    return span[0]


def find_relation(ent_a: Span, ent_b: Span) -> Optional[tuple[str, str]]:
    """
    Attempt to find a dependency-based relation between two entity spans
    that appear in the same sentence.

    Returns (relation_label, direction) or None.
    Examples:
        "Miles Davis recorded Kind of Blue" →
            subject=Miles Davis, verb=recorded, object=Kind of Blue
            → ("recorded", "subject→object")
    """
    root_a = token_root(ent_a)
    root_b = token_root(ent_b)

    # Check if they share a common verbal head
    head_a = root_a.head
    head_b = root_b.head

    if head_a == head_b and head_a.pos_ == "VERB":
        verb = head_a.lemma_
        dep_a = root_a.dep_
        dep_b = root_b.dep_
        if dep_a in SUBJECT_DEPS and dep_b in OBJECT_DEPS:
            return (verb, "subject→object")
        if dep_b in SUBJECT_DEPS and dep_a in OBJECT_DEPS:
            return (verb, "object→subject")
        return (verb, f"{dep_a}↔{dep_b}")

    # Prepositional patterns: "born in New Orleans", "member of Blue Note"
    if root_b.dep_ == "pobj" and root_b.head.dep_ in {"prep", "agent"}:
        prep = root_b.head.text.lower()
        prep_head = root_b.head.head
        if prep_head in ent_a or prep_head == root_a:
            return (prep, "head→pobj")

    if root_a.dep_ == "pobj" and root_a.head.dep_ in {"prep", "agent"}:
        prep = root_a.head.text.lower()
        prep_head = root_a.head.head
        if prep_head in ent_b or prep_head == root_b:
            return (prep, "head→pobj")

    return None


# ---------------------------------------------------------------------------
# Core NER pipeline
# ---------------------------------------------------------------------------

class NERPipeline:
    """Load spaCy model once, process all documents, extract entities + relations."""

    def __init__(
        self,
        input_path: Path = INPUT_PATH,
        output_path: Path = OUTPUT_PATH,
        model: str = SPACY_MODEL,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model
        self.nlp: Optional[spacy.Language] = None

    def load_model(self) -> None:
        logger.info("Loading spaCy model: %s", self.model_name)
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            logger.error(
                "Model '%s' not found. Run: python -m spacy download %s",
                self.model_name,
                self.model_name,
            )
            raise
        # Disable components we don't need to speed things up a little
        # (NER and parser are kept; tagger is also kept for POS used in relations)
        logger.info("Model loaded successfully.")

    def run(self) -> None:
        if self.nlp is None:
            self.load_model()

        if not self.input_path.exists():
            logger.error("Input file not found: %s", self.input_path)
            raise FileNotFoundError(self.input_path)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        doc_count = 0
        entity_count = 0

        with open(self.input_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSONL line: %s", exc)
                    continue

                url = record.get("url", "")
                text = record.get("text", "")
                if not text:
                    continue

                text = clean_text(text)
                doc_count += 1
                logger.info("[%d] Processing: %s", doc_count, url)

                page_rows = self._process_document(text, url)
                rows.extend(page_rows)
                entity_count += len(page_rows)
                logger.info("  Extracted %d entity rows from this page", len(page_rows))

        logger.info(
            "Finished processing %d documents. Total rows: %d",
            doc_count,
            entity_count,
        )
        self._write_csv(rows)

    def _process_document(self, text: str, url: str) -> list[dict]:
        """
        Run NER + dependency parsing on a document text, returning a list
        of row dicts ready for CSV output.
        """
        rows: list[dict] = []
        chunks = split_into_chunks(text)

        for chunk in chunks:
            try:
                doc: Doc = self.nlp(chunk)
            except Exception as exc:
                logger.warning("spaCy processing error: %s", exc)
                continue

            for sent in doc.sents:
                if len(sent) > MAX_SENTENCE_TOKENS:
                    continue  # skip very long sentences

                sent_text = sent.text.strip()
                if not sent_text:
                    continue

                # Collect target entities in this sentence
                sent_ents = [
                    ent for ent in sent.ents
                    if ent.label_ in TARGET_ENT_TYPES
                    and len(ent.text.strip()) >= MIN_ENTITY_LENGTH
                ]

                if not sent_ents:
                    continue

                # For each entity, emit a base row
                for ent in sent_ents:
                    base_row = {
                        "entity": ent.text.strip(),
                        "entity_type": ent.label_,
                        "source_url": url,
                        "context_sentence": sent_text,
                        "relation_to": "",
                        "relation_type": "",
                    }
                    rows.append(base_row)

                # Attempt pairwise relation extraction between entities in same sentence
                if len(sent_ents) >= 2:
                    relation_rows = self._extract_relations(sent_ents, url, sent_text)
                    rows.extend(relation_rows)

        return rows

    def _extract_relations(
        self, entities: list[Span], url: str, sent_text: str
    ) -> list[dict]:
        """
        Given a list of entities in the same sentence, try to extract
        dependency-based relations between pairs.
        Returns rows only for pairs where a relation was found.
        """
        relation_rows: list[dict] = []
        for i, ent_a in enumerate(entities):
            for ent_b in entities[i + 1:]:
                # Skip trivially same entity
                if ent_a.text.strip().lower() == ent_b.text.strip().lower():
                    continue
                relation = find_relation(ent_a, ent_b)
                if relation is not None:
                    verb, direction = relation
                    # Direction determines which entity is the "subject"
                    if "subject→object" in direction or "head→pobj" in direction:
                        subj, obj = ent_a, ent_b
                    elif "object→subject" in direction:
                        subj, obj = ent_b, ent_a
                    else:
                        subj, obj = ent_a, ent_b

                    relation_rows.append({
                        "entity": subj.text.strip(),
                        "entity_type": subj.label_,
                        "source_url": url,
                        "context_sentence": sent_text,
                        "relation_to": obj.text.strip(),
                        "relation_type": verb,
                    })
        return relation_rows

    def _write_csv(self, rows: list[dict]) -> None:
        """Write extracted rows to CSV."""
        # Deduplicate: same (entity, entity_type, relation_to, source_url)
        seen: set[tuple] = set()
        deduped: list[dict] = []
        for row in rows:
            key = (
                row["entity"].lower(),
                row["entity_type"],
                row["relation_to"].lower(),
                row["source_url"],
            )
            if key not in seen:
                seen.add(key)
                deduped.append(row)

        with open(self.output_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(deduped)

        logger.info(
            "Wrote %d rows (%d after dedup) to %s",
            len(rows),
            len(deduped),
            self.output_path,
        )


# ---------------------------------------------------------------------------
# Summary statistics helper
# ---------------------------------------------------------------------------

def print_summary(output_path: Path) -> None:
    """Print a quick breakdown of entity types found."""
    from collections import Counter
    import csv as csv_mod

    if not output_path.exists():
        logger.error("Output CSV not found: %s", output_path)
        return

    type_counter: Counter = Counter()
    relation_count = 0
    total = 0

    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            total += 1
            type_counter[row["entity_type"]] += 1
            if row["relation_to"]:
                relation_count += 1

    logger.info("--- NER Summary ---")
    logger.info("Total rows: %d", total)
    logger.info("Rows with relations: %d", relation_count)
    for etype, cnt in type_counter.most_common():
        logger.info("  %-15s: %d", etype, cnt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline = NERPipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        model=SPACY_MODEL,
    )
    pipeline.run()
    print_summary(OUTPUT_PATH)


if __name__ == "__main__":
    main()
