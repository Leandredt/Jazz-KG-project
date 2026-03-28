"""
Lab 2 - Jazz Ontology Builder
================================
Creates an OWL ontology for the Jazz Knowledge Graph domain.

Usage:
    python src/kg/ontology.py

Output:
    kg_artifacts/ontology.ttl
"""

from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal, URIRef

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ONTOLOGY_PATH = PROJECT_ROOT / "kg_artifacts" / "ontology.ttl"

JAZZ = Namespace("http://jazz-kg.org/ontology#")
SCHEMA = Namespace("https://schema.org/")


def build_ontology() -> Graph:
    """Build the Jazz OWL ontology with classes and properties."""
    g = Graph()
    g.bind("jazz", JAZZ)
    g.bind("schema", SCHEMA)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # Ontology declaration
    onto_uri = URIRef("http://jazz-kg.org/ontology")
    g.add((onto_uri, RDF.type, OWL.Ontology))
    g.add((onto_uri, RDFS.label, Literal("Jazz Music Knowledge Graph Ontology")))
    g.add((onto_uri, RDFS.comment, Literal("OWL ontology for representing jazz musicians, albums, labels, bands, and locations.")))

    # ------------------------------------------------------------------ #
    # Classes                                                              #
    # ------------------------------------------------------------------ #
    classes = {
        JAZZ.Musician: ("Musician", "A jazz musician or performer.", SCHEMA.Person),
        JAZZ.Album: ("Album", "A recorded jazz album.", SCHEMA.MusicAlbum),
        JAZZ.RecordLabel: ("RecordLabel", "A music record label.", SCHEMA.Organization),
        JAZZ.Band: ("Band", "A jazz band or ensemble.", SCHEMA.MusicGroup),
        JAZZ.Location: ("Location", "A geographic location associated with jazz.", SCHEMA.Place),
        JAZZ.Instrument: ("Instrument", "A musical instrument played in jazz.", SCHEMA.Thing),
        JAZZ.Genre: ("Genre", "A jazz sub-genre or style.", SCHEMA.MusicGenre),
    }
    for cls_uri, (label, comment, equiv) in classes.items():
        g.add((cls_uri, RDF.type, OWL.Class))
        g.add((cls_uri, RDFS.label, Literal(label)))
        g.add((cls_uri, RDFS.comment, Literal(comment)))
        g.add((cls_uri, OWL.equivalentClass, equiv))

    # ------------------------------------------------------------------ #
    # Object Properties                                                    #
    # ------------------------------------------------------------------ #
    obj_props = {
        JAZZ.playedBy: ("playedBy", "Links an album to the musician who performed it.",
                        JAZZ.Album, JAZZ.Musician),
        JAZZ.releasedBy: ("releasedBy", "Links an album to its record label.",
                          JAZZ.Album, JAZZ.RecordLabel),
        JAZZ.memberOf: ("memberOf", "Links a musician to a band they belong to.",
                        JAZZ.Musician, JAZZ.Band),
        JAZZ.bornIn: ("bornIn", "Links a musician to their birth location.",
                      JAZZ.Musician, JAZZ.Location),
        JAZZ.hasGenre: ("hasGenre", "Links a musician or album to a jazz genre.",
                        None, JAZZ.Genre),
        JAZZ.recordedOn: ("recordedOn", "Links a musician to an album they recorded.",
                          JAZZ.Musician, JAZZ.Album),
        JAZZ.signedWith: ("signedWith", "Links a musician to a record label (inferred by SWRL).",
                          JAZZ.Musician, JAZZ.RecordLabel),
        JAZZ.basedIn: ("basedIn", "Links a band or label to a location.",
                       None, JAZZ.Location),
        JAZZ.plays: ("plays", "Links a musician to an instrument they play.",
                     JAZZ.Musician, JAZZ.Instrument),
        JAZZ.influencedBy: ("influencedBy", "Links a musician to another who influenced them.",
                            JAZZ.Musician, JAZZ.Musician),
    }
    for prop_uri, (label, comment, domain, range_cls) in obj_props.items():
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDFS.label, Literal(label)))
        g.add((prop_uri, RDFS.comment, Literal(comment)))
        if domain:
            g.add((prop_uri, RDFS.domain, domain))
        g.add((prop_uri, RDFS.range, range_cls))

    # ------------------------------------------------------------------ #
    # Datatype Properties                                                  #
    # ------------------------------------------------------------------ #
    dt_props = {
        JAZZ.birthYear: ("birthYear", "Year of birth of a musician.", JAZZ.Musician, XSD.integer),
        JAZZ.deathYear: ("deathYear", "Year of death (if applicable).", JAZZ.Musician, XSD.integer),
        JAZZ.releaseYear: ("releaseYear", "Year an album was released.", JAZZ.Album, XSD.integer),
        JAZZ.wikidataId: ("wikidataId", "Wikidata QID for entity alignment.", None, XSD.string),
        JAZZ.confidence: ("confidence", "Confidence score from entity linking (0.0–1.0).", None, XSD.float),
    }
    for prop_uri, (label, comment, domain, range_cls) in dt_props.items():
        g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        g.add((prop_uri, RDFS.label, Literal(label)))
        g.add((prop_uri, RDFS.comment, Literal(comment)))
        if domain:
            g.add((prop_uri, RDFS.domain, domain))
        g.add((prop_uri, RDFS.range, range_cls))

    # ------------------------------------------------------------------ #
    # Predicate alignment: jazz → Wikidata                                 #
    # ------------------------------------------------------------------ #
    WDT = Namespace("http://www.wikidata.org/prop/direct/")
    g.bind("wdt", WDT)
    alignments = [
        (JAZZ.playedBy, WDT.P175),      # performer
        (JAZZ.releasedBy, WDT.P264),    # record label
        (JAZZ.memberOf, WDT.P463),      # member of
        (JAZZ.bornIn, WDT.P19),         # place of birth
        (JAZZ.hasGenre, WDT.P136),      # genre
        (JAZZ.plays, WDT.P1303),        # instrument
        (JAZZ.influencedBy, WDT.P737),  # influenced by
    ]
    for jazz_prop, wd_prop in alignments:
        g.add((jazz_prop, OWL.equivalentProperty, wd_prop))

    return g


def save_ontology(g: Graph, path: Path = ONTOLOGY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(path), format="turtle")
    print(f"Ontology saved → {path}  ({len(g)} triples)")


def main() -> None:
    g = build_ontology()
    save_ontology(g)


if __name__ == "__main__":
    main()
