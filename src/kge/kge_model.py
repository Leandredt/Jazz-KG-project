"""
Knowledge Graph Embeddings (KGE) Module — TransE implementation
===============================================================
Trains a TransE-like embedding model on an RDF graph using numpy.
Falls back gracefully if PyKEEN is unavailable.

Interface:
    KGEModel(kg: Graph, output_dir: Path)
    .train_embeddings()  — trains and persists artifacts
    .get_embeddings()    — returns {entities: {...}, relations: {...}}
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
from rdflib import Graph, URIRef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triples_from_graph(g: Graph) -> list[tuple[str, str, str]]:
    """Extract (subject, predicate, object) string triples from an RDF graph."""
    triples = []
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
            triples.append((str(s), str(p), str(o)))
    return triples


def _build_mappings(triples: list[tuple[str, str, str]]) -> tuple[dict, dict]:
    """Build entity2id and relation2id mappings from a list of triples."""
    entities: set[str] = set()
    relations: set[str] = set()
    for s, p, o in triples:
        entities.add(s)
        entities.add(o)
        relations.add(p)
    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    return entity2id, relation2id


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (in-place safe)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


# ---------------------------------------------------------------------------
# TransE trainer (numpy)
# ---------------------------------------------------------------------------

class _TransENumpy:
    """Minimal TransE trainer using numpy only."""

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        dim: int = 50,
        margin: float = 1.0,
        lr: float = 0.01,
        n_epochs: int = 20,
        batch_size: int = 128,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        # Entity and relation embeddings
        self.ent_emb = _l2_normalize(
            rng.uniform(-0.1, 0.1, size=(n_entities, dim)).astype(np.float32)
        )
        self.rel_emb = _l2_normalize(
            rng.uniform(-0.1, 0.1, size=(n_relations, dim)).astype(np.float32)
        )
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.margin = margin
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rng = rng

    def _score(self, h: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
        """TransE score: -||h + r - t||_2  (lower = better positive)."""
        return -np.linalg.norm(h + r - t, axis=1)

    def train(self, train_ids: list[tuple[int, int, int]]) -> None:
        """Run TransE training for self.n_epochs epochs."""
        n = len(train_ids)
        train_arr = np.array(train_ids, dtype=np.int32)
        logger.info("TransE training: %d triples, dim=%d, epochs=%d", n, self.dim, self.n_epochs)

        for epoch in range(self.n_epochs):
            # Shuffle
            idx = self.rng.permutation(n)
            train_arr = train_arr[idx]
            total_loss = 0.0

            for start in range(0, n, self.batch_size):
                batch = train_arr[start: start + self.batch_size]
                if len(batch) == 0:
                    continue

                h_ids = batch[:, 0]
                r_ids = batch[:, 1]
                t_ids = batch[:, 2]

                # Corrupt tails (50%) or heads (50%) for negatives
                neg_ids = self.rng.integers(0, self.n_entities, size=len(batch))
                corrupt_tail = self.rng.integers(0, 2, size=len(batch)).astype(bool)
                nh_ids = np.where(corrupt_tail, h_ids, neg_ids)
                nt_ids = np.where(corrupt_tail, neg_ids, t_ids)

                h = self.ent_emb[h_ids]
                r = self.rel_emb[r_ids]
                t = self.ent_emb[t_ids]
                nh = self.ent_emb[nh_ids]
                nt = self.ent_emb[nt_ids]

                pos_dist = np.linalg.norm(h + r - t, axis=1)
                neg_dist = np.linalg.norm(nh + r - nt, axis=1)
                loss = np.maximum(0.0, self.margin + pos_dist - neg_dist)
                total_loss += loss.sum()

                # Compute gradients via sub-gradient
                active = (loss > 0).astype(np.float32)[:, None]

                # Positive gradient
                diff_pos = h + r - t
                sign_pos = np.sign(diff_pos)

                # Negative gradient
                diff_neg = nh + r - nt
                sign_neg = np.sign(diff_neg)

                grad_h = active * sign_pos
                grad_t = -active * sign_pos
                grad_r = active * (sign_pos - sign_neg)
                grad_nh = -active * sign_neg
                grad_nt = active * sign_neg

                # Apply updates (sum over batch for shared indices)
                np.add.at(self.ent_emb, h_ids, -self.lr * grad_h)
                np.add.at(self.ent_emb, t_ids, -self.lr * grad_t)
                np.add.at(self.rel_emb, r_ids, -self.lr * grad_r)
                np.add.at(self.ent_emb, nh_ids, -self.lr * grad_nh)
                np.add.at(self.ent_emb, nt_ids, -self.lr * grad_nt)

                # Re-normalise entities
                self.ent_emb = _l2_normalize(self.ent_emb)

            if (epoch + 1) % 5 == 0:
                logger.info("  Epoch %d/%d — loss=%.2f", epoch + 1, self.n_epochs, total_loss)

        logger.info("TransE training complete.")


# ---------------------------------------------------------------------------
# DistMult trainer (numpy)
# ---------------------------------------------------------------------------

class _DistMultNumpy:
    """Minimal DistMult trainer using numpy only.

    Score: sigma(h * r * t) where * is element-wise product.
    Training uses sampled softmax-style loss with negative sampling.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        dim: int = 50,
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 128,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(dim)
        self.ent_emb = rng.uniform(-scale, scale, size=(n_entities, dim)).astype(np.float32)
        self.rel_emb = rng.uniform(-scale, scale, size=(n_relations, dim)).astype(np.float32)
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rng = rng

    def _score(self, h: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
        """DistMult score: sum(h * r * t)  (higher = better)."""
        return np.sum(h * r * t, axis=1)

    def train(self, train_ids: list[tuple[int, int, int]]) -> None:
        """Run DistMult training using margin-based loss with negative sampling."""
        n = len(train_ids)
        train_arr = np.array(train_ids, dtype=np.int32)
        logger.info("DistMult training: %d triples, dim=%d, epochs=%d", n, self.dim, self.n_epochs)

        margin = 1.0

        for epoch in range(self.n_epochs):
            idx = self.rng.permutation(n)
            train_arr = train_arr[idx]
            total_loss = 0.0

            for start in range(0, n, self.batch_size):
                batch = train_arr[start: start + self.batch_size]
                if len(batch) == 0:
                    continue

                h_ids = batch[:, 0]
                r_ids = batch[:, 1]
                t_ids = batch[:, 2]

                # Corrupt tails (50%) or heads (50%)
                neg_ids = self.rng.integers(0, self.n_entities, size=len(batch))
                corrupt_tail = self.rng.integers(0, 2, size=len(batch)).astype(bool)
                nh_ids = np.where(corrupt_tail, h_ids, neg_ids)
                nt_ids = np.where(corrupt_tail, neg_ids, t_ids)

                h = self.ent_emb[h_ids]
                r = self.rel_emb[r_ids]
                t = self.ent_emb[t_ids]
                nh = self.ent_emb[nh_ids]
                nt = self.ent_emb[nt_ids]

                pos_score = self._score(h, r, t)
                neg_score = self._score(nh, r, nt)

                # Margin loss: max(0, margin - pos + neg)
                loss_vec = np.maximum(0.0, margin - pos_score + neg_score)
                total_loss += loss_vec.sum()

                active = (loss_vec > 0).astype(np.float32)[:, None]

                # Gradients of DistMult score w.r.t. embeddings
                # d(score_pos)/d(h) = r * t, etc.
                grad_h_pos  =  active * (r * t)
                grad_t_pos  =  active * (h * r)
                grad_r_pos  =  active * (h * t)
                grad_nh_neg =  active * (r * nt)
                grad_nt_neg =  active * (nh * r)
                grad_r_neg  =  active * (nh * nt)

                # Update: maximise pos_score, minimise neg_score
                np.add.at(self.ent_emb, h_ids,  self.lr * grad_h_pos)
                np.add.at(self.ent_emb, t_ids,  self.lr * grad_t_pos)
                np.add.at(self.rel_emb, r_ids,  self.lr * (grad_r_pos - grad_r_neg))
                np.add.at(self.ent_emb, nh_ids, -self.lr * grad_nh_neg)
                np.add.at(self.ent_emb, nt_ids, -self.lr * grad_nt_neg)

            if (epoch + 1) % 10 == 0:
                logger.info("  Epoch %d/%d — loss=%.2f", epoch + 1, self.n_epochs, total_loss)

        logger.info("DistMult training complete.")


# ---------------------------------------------------------------------------
# KGEModel — public interface
# ---------------------------------------------------------------------------

class KGEModel:
    """
    Knowledge Graph Embedding model with a TransE backend (numpy).

    Parameters
    ----------
    kg : rdflib.Graph
        The knowledge graph to embed.
    output_dir : Path
        Directory where artifacts (txt splits, mappings, embeddings) are saved.
    dim : int
        Embedding dimensionality (default 50).
    n_epochs : int
        Number of training epochs (default 20).
    """

    def __init__(self, kg: Graph, output_dir: Path, dim: int = 50, n_epochs: int = 20):
        self.kg = kg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.n_epochs = n_epochs

        self._entity2id: Optional[dict[str, int]] = None
        self._relation2id: Optional[dict[str, int]] = None
        self._ent_emb: Optional[np.ndarray] = None
        self._rel_emb: Optional[np.ndarray] = None

        logger.info("KGEModel initialised — output_dir=%s, dim=%d", self.output_dir, self.dim)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train_embeddings(self) -> None:
        """
        Extract triples from the KG, split 80/10/10, train TransE,
        and persist all artifacts.
        """
        logger.info("Extracting triples from KG (%d total)…", len(self.kg))
        triples = _triples_from_graph(self.kg)

        if not triples:
            logger.warning("No URI-only triples found in KG; skipping training.")
            return

        logger.info("Found %d URI triples for KGE.", len(triples))

        # Build id mappings
        entity2id, relation2id = _build_mappings(triples)
        self._entity2id = entity2id
        self._relation2id = relation2id

        # Save mappings
        with open(self.output_dir / "entity2id.json", "w", encoding="utf-8") as f:
            json.dump(entity2id, f, ensure_ascii=False, indent=2)
        with open(self.output_dir / "relation2id.json", "w", encoding="utf-8") as f:
            json.dump(relation2id, f, ensure_ascii=False, indent=2)
        logger.info("Saved entity2id (%d) and relation2id (%d).", len(entity2id), len(relation2id))

        # Convert to id triples
        id_triples = [
            (entity2id[s], relation2id[p], entity2id[o])
            for s, p, o in triples
        ]

        # Split 80/10/10
        random.seed(42)
        random.shuffle(id_triples)
        n = len(id_triples)
        n_train = max(1, int(n * 0.8))
        n_valid = max(1, int(n * 0.1))
        train_ids = id_triples[:n_train]
        valid_ids = id_triples[n_train: n_train + n_valid]
        test_ids = id_triples[n_train + n_valid:]

        # Save txt splits (tab-separated h r t)
        for split_name, split_data in [("train", train_ids), ("valid", valid_ids), ("test", test_ids)]:
            path = self.output_dir / f"{split_name}.txt"
            with open(path, "w", encoding="utf-8") as f:
                for h, r, t in split_data:
                    f.write(f"{h}\t{r}\t{t}\n")
        logger.info("Saved splits: train=%d, valid=%d, test=%d", len(train_ids), len(valid_ids), len(test_ids))

        # Train TransE
        n_ent = len(entity2id)
        n_rel = len(relation2id)
        model = _TransENumpy(
            n_entities=n_ent,
            n_relations=n_rel,
            dim=self.dim,
            n_epochs=self.n_epochs,
        )
        model.train(train_ids)

        self._ent_emb = model.ent_emb
        self._rel_emb = model.rel_emb

        # Save embeddings
        emb_path = self.output_dir / "embeddings.npz"
        np.savez(emb_path, entity_embeddings=self._ent_emb, relation_embeddings=self._rel_emb)
        logger.info("Embeddings saved → %s  (entities=%s, relations=%s)", emb_path, self._ent_emb.shape, self._rel_emb.shape)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_embeddings(self) -> dict:
        """
        Return entity and relation embeddings as dicts mapping URI → vector.
        If not yet trained, attempts to load from disk.
        """
        if self._ent_emb is None:
            self._load_from_disk()

        if self._ent_emb is None or self._entity2id is None:
            logger.warning("No embeddings available — run train_embeddings() first.")
            return {"entities": {}, "relations": {}}

        id2entity = {v: k for k, v in self._entity2id.items()}
        id2relation = {v: k for k, v in self._relation2id.items()}

        return {
            "entities": {id2entity[i]: self._ent_emb[i].tolist() for i in range(len(id2entity))},
            "relations": {id2relation[i]: self._rel_emb[i].tolist() for i in range(len(id2relation))},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Attempt to load saved embeddings and mappings from output_dir."""
        emb_path = self.output_dir / "embeddings.npz"
        e2id_path = self.output_dir / "entity2id.json"
        r2id_path = self.output_dir / "relation2id.json"

        if emb_path.exists() and e2id_path.exists() and r2id_path.exists():
            data = np.load(emb_path)
            self._ent_emb = data["entity_embeddings"]
            self._rel_emb = data["relation_embeddings"]
            with open(e2id_path, encoding="utf-8") as f:
                self._entity2id = json.load(f)
            with open(r2id_path, encoding="utf-8") as f:
                self._relation2id = json.load(f)
            logger.info("Loaded embeddings from %s", emb_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    expanded_path = PROJECT_ROOT / "kg_artifacts" / "expanded.nt"
    output_dir = PROJECT_ROOT / "kge_artifacts"

    g = Graph()
    if expanded_path.exists():
        logger.info("Loading expanded KG from %s…", expanded_path)
        g.parse(str(expanded_path), format="nt")
    else:
        initial_path = PROJECT_ROOT / "kg_artifacts" / "initial_kg.ttl"
        if initial_path.exists():
            logger.info("expanded.nt not found; loading initial_kg.ttl instead…")
            g.parse(str(initial_path), format="turtle")
        else:
            logger.error("No KG file found. Run the pipeline first.")
            sys.exit(1)

    model = KGEModel(g, output_dir)
    model.train_embeddings()
    embeddings = model.get_embeddings()
    n_ent = len(embeddings["entities"])
    n_rel = len(embeddings["relations"])
    print(f"Done. {n_ent} entity embeddings, {n_rel} relation embeddings.")
