"""
Knowledge Graph Reasoner — SWRL-like rule engine
=================================================
Applies forward-chaining rules to infer new RDF triples from existing data,
and validates KG consistency.

Rules implemented
-----------------
1. signedWith  : ?album jazz:releasedBy ?label  ∧  ?musician jazz:recordedOn ?album
                 → ?musician jazz:signedWith ?label
2. bornIn inheritance: ?musician jazz:memberOf ?band  ∧  ?band jazz:basedIn ?place
                 → ?musician jazz:bornIn ?place   (if not already asserted)
3. influencedBy transitivity (2 hops):
                 ?a jazz:influencedBy ?b  ∧  ?b jazz:influencedBy ?c
                 → ?a jazz:influencedBy ?c
4. subClassOf RDFS closure:
                 ?instance rdf:type ?subclass  ∧  ?subclass rdfs:subClassOf ?superclass
                 → ?instance rdf:type ?superclass

Consistency checks
------------------
- No entity can be both jazz:Musician and jazz:RecordLabel.
- No entity can be both jazz:Album and jazz:Band.
- rdfs:label should be a Literal, not a URIRef.
"""

import logging
from pathlib import Path

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal

logger = logging.getLogger(__name__)

JAZZ = Namespace("http://jazz-kg.org/ontology#")


class Reasoner:
    """
    Forward-chaining reasoner for the Jazz KG.

    Parameters
    ----------
    kg : rdflib.Graph
        The graph to reason over.  New triples are added in-place.
    """

    def __init__(self, kg: Graph):
        self.kg = kg
        logger.info("Reasoner initialised — KG has %d triples.", len(kg))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer_new_facts(self) -> int:
        """
        Run all inference rules and return the total count of new triples added.
        """
        before = len(self.kg)

        added = 0
        added += self._rule_signed_with()
        added += self._rule_born_in_from_band()
        added += self._rule_influenced_by_transitive()
        added += self._rule_rdfs_subclass_closure()

        after = len(self.kg)
        actual = after - before
        logger.info("Inference complete: %d new triples added (rules reported %d).", actual, added)
        return actual

    def validate_consistency(self) -> bool:
        """
        Check the KG for type-level contradictions.
        Returns True if no violations are found.
        """
        logger.info("Validating KG consistency…")
        violations: list[str] = []

        # 1. An entity cannot be both Musician and RecordLabel
        musicians = set(self.kg.subjects(RDF.type, JAZZ.Musician))
        labels = set(self.kg.subjects(RDF.type, JAZZ.RecordLabel))
        both_ml = musicians & labels
        for uri in both_ml:
            msg = f"Contradiction: {uri} is both Musician and RecordLabel"
            logger.warning(msg)
            violations.append(msg)

        # 2. An entity cannot be both Album and Band
        albums = set(self.kg.subjects(RDF.type, JAZZ.Album))
        bands = set(self.kg.subjects(RDF.type, JAZZ.Band))
        both_ab = albums & bands
        for uri in both_ab:
            msg = f"Contradiction: {uri} is both Album and Band"
            logger.warning(msg)
            violations.append(msg)

        # 3. rdfs:label values should be Literals
        for s, _, o in self.kg.triples((None, RDFS.label, None)):
            if isinstance(o, URIRef):
                msg = f"rdfs:label of {s} is a URIRef, expected Literal"
                logger.warning(msg)
                violations.append(msg)

        if violations:
            logger.error("Consistency check FAILED: %d violation(s).", len(violations))
            return False

        logger.info("Consistency check PASSED — no violations found.")
        return True

    # ------------------------------------------------------------------
    # Rule 1: signedWith
    # ------------------------------------------------------------------

    def _rule_signed_with(self) -> int:
        """
        ?album jazz:releasedBy ?label  ∧  ?musician jazz:recordedOn ?album
        → ?musician jazz:signedWith ?label
        """
        new_triples: list[tuple] = []

        # Index albums → labels
        album_to_label: dict[URIRef, URIRef] = {}
        for album, _, label in self.kg.triples((None, JAZZ.releasedBy, None)):
            if isinstance(album, URIRef) and isinstance(label, URIRef):
                album_to_label[album] = label

        for musician, _, album in self.kg.triples((None, JAZZ.recordedOn, None)):
            if not isinstance(musician, URIRef) or not isinstance(album, URIRef):
                continue
            label_uri = album_to_label.get(album)
            if label_uri is None:
                continue
            triple = (musician, JAZZ.signedWith, label_uri)
            if triple not in self.kg:
                new_triples.append(triple)

        for t in new_triples:
            self.kg.add(t)
        if new_triples:
            logger.info("Rule signedWith: inferred %d triples.", len(new_triples))
        return len(new_triples)

    # ------------------------------------------------------------------
    # Rule 2: bornIn from band basedIn
    # ------------------------------------------------------------------

    def _rule_born_in_from_band(self) -> int:
        """
        ?musician jazz:memberOf ?band  ∧  ?band jazz:basedIn ?place
        → ?musician jazz:bornIn ?place   (only if musician has no bornIn yet)
        """
        new_triples: list[tuple] = []

        # Index band → place
        band_to_place: dict[URIRef, URIRef] = {}
        for band, _, place in self.kg.triples((None, JAZZ.basedIn, None)):
            if isinstance(band, URIRef) and isinstance(place, URIRef):
                band_to_place[band] = place

        # Musicians already with a bornIn
        already_born: set[URIRef] = set(self.kg.subjects(JAZZ.bornIn, None))

        for musician, _, band in self.kg.triples((None, JAZZ.memberOf, None)):
            if not isinstance(musician, URIRef) or not isinstance(band, URIRef):
                continue
            if musician in already_born:
                continue
            place = band_to_place.get(band)
            if place is None:
                continue
            triple = (musician, JAZZ.bornIn, place)
            if triple not in self.kg:
                new_triples.append(triple)

        for t in new_triples:
            self.kg.add(t)
        if new_triples:
            logger.info("Rule bornIn-from-band: inferred %d triples.", len(new_triples))
        return len(new_triples)

    # ------------------------------------------------------------------
    # Rule 3: influencedBy transitivity (2 hops)
    # ------------------------------------------------------------------

    def _rule_influenced_by_transitive(self) -> int:
        """
        ?a jazz:influencedBy ?b  ∧  ?b jazz:influencedBy ?c
        → ?a jazz:influencedBy ?c
        """
        new_triples: list[tuple] = []

        # Build direct influence map
        influences: dict[URIRef, set[URIRef]] = {}
        for a, _, b in self.kg.triples((None, JAZZ.influencedBy, None)):
            if isinstance(a, URIRef) and isinstance(b, URIRef):
                influences.setdefault(a, set()).add(b)

        for a, direct in influences.items():
            for b in list(direct):
                for c in influences.get(b, set()):
                    if c == a:
                        continue  # avoid self-loops
                    triple = (a, JAZZ.influencedBy, c)
                    if triple not in self.kg:
                        new_triples.append(triple)

        for t in new_triples:
            self.kg.add(t)
        if new_triples:
            logger.info("Rule influencedBy-transitive: inferred %d triples.", len(new_triples))
        return len(new_triples)

    # ------------------------------------------------------------------
    # Rule 4: RDFS subClassOf closure
    # ------------------------------------------------------------------

    def _rule_rdfs_subclass_closure(self) -> int:
        """
        ?instance rdf:type ?sub  ∧  ?sub rdfs:subClassOf ?super
        → ?instance rdf:type ?super
        """
        new_triples: list[tuple] = []

        # Build subclass map
        subclass_of: dict[URIRef, set[URIRef]] = {}
        for sub, _, sup in self.kg.triples((None, RDFS.subClassOf, None)):
            if isinstance(sub, URIRef) and isinstance(sup, URIRef):
                subclass_of.setdefault(sub, set()).add(sup)

        if not subclass_of:
            return 0

        for instance, _, sub in self.kg.triples((None, RDF.type, None)):
            if not isinstance(instance, URIRef) or not isinstance(sub, URIRef):
                continue
            for sup in subclass_of.get(sub, set()):
                triple = (instance, RDF.type, sup)
                if triple not in self.kg:
                    new_triples.append(triple)

        for t in new_triples:
            self.kg.add(t)
        if new_triples:
            logger.info("Rule rdfs-subclass-closure: inferred %d triples.", len(new_triples))
        return len(new_triples)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    kg_path = PROJECT_ROOT / "kg_artifacts" / "initial_kg.ttl"

    g = Graph()
    if kg_path.exists():
        g.parse(str(kg_path), format="turtle")
        logger.info("Loaded KG: %d triples", len(g))
    else:
        logger.error("No KG found at %s. Run build_kg.py first.", kg_path)
        sys.exit(1)

    reasoner = Reasoner(g)
    inferred = reasoner.infer_new_facts()
    consistent = reasoner.validate_consistency()
    print(f"\nInferred {inferred} new facts.")
    print(f"KG consistency: {'OK' if consistent else 'VIOLATIONS FOUND'}")
    print(f"Total triples after reasoning: {len(g)}")
