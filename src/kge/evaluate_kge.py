"""
KGE Evaluation
==============
Loads train/valid/test splits, trains TransE and DistMult, evaluates
MRR / Hits@1 / Hits@3 / Hits@10, and produces a t-SNE (or PCA) plot.
"""

import json
import logging
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KGE_DATA_DIR = PROJECT_ROOT / "data" / "kge"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_split(path: Path) -> list[tuple[int, int, int]]:
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return triples


def _load_mapping(path: Path) -> dict:
    """Load entity2id.txt or relation2id.txt → {uri: id}."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    # First line is count
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            mapping[parts[0]] = int(parts[1])
    return mapping


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


# ---------------------------------------------------------------------------
# TransE (100 epochs)
# ---------------------------------------------------------------------------

class TransEModel:
    def __init__(self, n_entities, n_relations, dim=50, margin=1.0, lr=0.01,
                 n_epochs=100, batch_size=256, seed=42):
        rng = np.random.default_rng(seed)
        self.ent_emb = _l2_normalize(
            rng.uniform(-0.1, 0.1, size=(n_entities, dim)).astype(np.float32)
        )
        self.rel_emb = _l2_normalize(
            rng.uniform(-0.1, 0.1, size=(n_relations, dim)).astype(np.float32)
        )
        self.n_entities = n_entities
        self.dim = dim
        self.margin = margin
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rng = rng

    def score(self, h: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Lower is better for TransE."""
        return np.linalg.norm(h + r - t, axis=1)

    def train(self, train_ids):
        n = len(train_ids)
        train_arr = np.array(train_ids, dtype=np.int32)
        print(f"  [TransE] Training on {n} triples, dim={self.dim}, epochs={self.n_epochs}")

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

                active = (loss > 0).astype(np.float32)[:, None]
                sign_pos = np.sign(h + r - t)
                sign_neg = np.sign(nh + r - nt)

                np.add.at(self.ent_emb, h_ids, -self.lr * active * sign_pos)
                np.add.at(self.ent_emb, t_ids,  self.lr * active * sign_pos)
                np.add.at(self.rel_emb, r_ids, -self.lr * active * (sign_pos - sign_neg))
                np.add.at(self.ent_emb, nh_ids,  self.lr * active * sign_neg)
                np.add.at(self.ent_emb, nt_ids, -self.lr * active * sign_neg)
                self.ent_emb = _l2_normalize(self.ent_emb)

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.n_epochs} — loss={total_loss:.2f}")

        print("  [TransE] Training complete.")


# ---------------------------------------------------------------------------
# DistMult (100 epochs)
# ---------------------------------------------------------------------------

class DistMultModel:
    def __init__(self, n_entities, n_relations, dim=50, lr=0.01,
                 n_epochs=100, batch_size=256, seed=42):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(dim)
        self.ent_emb = rng.uniform(-scale, scale, size=(n_entities, dim)).astype(np.float32)
        self.rel_emb = rng.uniform(-scale, scale, size=(n_relations, dim)).astype(np.float32)
        self.n_entities = n_entities
        self.dim = dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rng = rng

    def score(self, h: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Higher is better for DistMult."""
        return np.sum(h * r * t, axis=1)

    def train(self, train_ids):
        n = len(train_ids)
        train_arr = np.array(train_ids, dtype=np.int32)
        print(f"  [DistMult] Training on {n} triples, dim={self.dim}, epochs={self.n_epochs}")
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

                neg_ids = self.rng.integers(0, self.n_entities, size=len(batch))
                corrupt_tail = self.rng.integers(0, 2, size=len(batch)).astype(bool)
                nh_ids = np.where(corrupt_tail, h_ids, neg_ids)
                nt_ids = np.where(corrupt_tail, neg_ids, t_ids)

                h = self.ent_emb[h_ids]
                r = self.rel_emb[r_ids]
                t = self.ent_emb[t_ids]
                nh = self.ent_emb[nh_ids]
                nt = self.ent_emb[nt_ids]

                pos_score = np.sum(h * r * t, axis=1)
                neg_score = np.sum(nh * r * nt, axis=1)
                loss_vec = np.maximum(0.0, margin - pos_score + neg_score)
                total_loss += loss_vec.sum()

                active = (loss_vec > 0).astype(np.float32)[:, None]

                np.add.at(self.ent_emb, h_ids,  self.lr * active * (r * t))
                np.add.at(self.ent_emb, t_ids,  self.lr * active * (h * r))
                np.add.at(self.rel_emb, r_ids,  self.lr * active * ((h * t) - (nh * nt)))
                np.add.at(self.ent_emb, nh_ids, -self.lr * active * (r * nt))
                np.add.at(self.ent_emb, nt_ids, -self.lr * active * (nh * r))

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.n_epochs} — loss={total_loss:.2f}")

        print("  [DistMult] Training complete.")


# ---------------------------------------------------------------------------
# Ranking evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, test_triples, all_triples_set, model_name="Model"):
    """
    For each test triple (h, r, t), rank the correct tail among all entities.
    Uses filtered ranking (removes other true tails from ranking).
    Returns MRR, Hits@1, Hits@3, Hits@10.
    """
    n_entities = model.n_entities
    all_ent_ids = np.arange(n_entities)

    # Build a set of all true tails per (h, r) for filtered ranking
    hr2tails: dict[tuple[int,int], set[int]] = {}
    for h, r, t in all_triples_set:
        hr2tails.setdefault((h, r), set()).add(t)

    ranks = []
    max_eval = min(len(test_triples), 2000)  # cap for speed
    sample = test_triples[:max_eval]

    print(f"  [{model_name}] Evaluating on {len(sample)} test triples …")

    # Pre-fetch all entity embeddings
    all_ent_emb = model.ent_emb  # shape (n_entities, dim)

    for i, (h_id, r_id, t_id) in enumerate(sample):
        h_emb = model.ent_emb[h_id]   # (dim,)
        r_emb = model.rel_emb[r_id]   # (dim,)

        # Broadcast: score all candidate tails
        H = np.tile(h_emb, (n_entities, 1))   # (n_ent, dim)
        R = np.tile(r_emb, (n_entities, 1))   # (n_ent, dim)
        T = all_ent_emb                         # (n_ent, dim)

        scores = model.score(H, R, T)  # (n_ent,)

        # Filtered setting: mask out other true tails
        true_tails = hr2tails.get((h_id, r_id), set())
        mask_ids = list(true_tails - {t_id})
        if mask_ids:
            # For TransE lower=better; for DistMult higher=better
            if hasattr(model, '_is_distance') or model_name == "TransE":
                scores[mask_ids] = np.inf  # push away
            else:
                scores[mask_ids] = -np.inf

        # Rank correct tail
        if model_name == "TransE":
            # Lower distance = better → ascending rank
            rank = int(np.sum(scores < scores[t_id])) + 1
        else:
            # Higher score = better → descending rank
            rank = int(np.sum(scores > scores[t_id])) + 1

        ranks.append(rank)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(sample)} evaluated …")

    ranks_arr = np.array(ranks, dtype=np.float32)
    mrr = float(np.mean(1.0 / ranks_arr))
    hits1 = float(np.mean(ranks_arr <= 1))
    hits3 = float(np.mean(ranks_arr <= 3))
    hits10 = float(np.mean(ranks_arr <= 10))

    return {"MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10}


# ---------------------------------------------------------------------------
# t-SNE / PCA visualization
# ---------------------------------------------------------------------------

def plot_tsne(ent_emb: np.ndarray, entity2id: dict, reports_dir: Path, n_sample: int = 200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_entities = ent_emb.shape[0]
    rng = np.random.default_rng(0)

    # Sample entities
    sampled_ids = rng.choice(n_entities, size=min(n_sample, n_entities), replace=False)
    sampled_emb = ent_emb[sampled_ids]  # (n_sample, dim)

    # Invert entity2id
    id2entity = {v: k for k, v in entity2id.items()}

    # Determine 2D projection
    try:
        from sklearn.manifold import TSNE
        print("  Using sklearn t-SNE for visualization …")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sampled_ids) - 1))
        coords = tsne.fit_transform(sampled_emb)
    except ImportError:
        print("  sklearn not available — using PCA (SVD) as fallback …")
        # Center
        mu = sampled_emb.mean(axis=0)
        X = sampled_emb - mu
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        coords = X @ Vt[:2].T  # project onto top-2 PCs

    # Assign colors by entity type from URI
    type_colors = {
        "Musician": "royalblue",
        "Album": "darkorange",
        "Location": "green",
        "Band": "red",
        "RecordLabel": "purple",
        "Instrument": "brown",
        "Genre": "pink",
        "Other": "grey",
    }

    def _get_type(uri: str) -> str:
        uri_lower = uri.lower()
        for t in ["musician", "album", "location", "band", "recordlabel", "instrument", "genre"]:
            if t in uri_lower:
                return t.capitalize() if t != "recordlabel" else "RecordLabel"
        # Check Wikidata URIs by prefix patterns — default to Other
        return "Other"

    colors = []
    labels_seen = set()
    for eid in sampled_ids:
        uri = id2entity.get(int(eid), "")
        t = _get_type(uri)
        colors.append(type_colors.get(t, "grey"))
        labels_seen.add(t)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot by type for legend
    plotted = set()
    for i, (eid, color) in enumerate(zip(sampled_ids, colors)):
        uri = id2entity.get(int(eid), "")
        t = _get_type(uri)
        label = t if t not in plotted else None
        plotted.add(t)
        ax.scatter(coords[i, 0], coords[i, 1], c=color, alpha=0.6, s=20, label=label)

    ax.set_title("t-SNE / PCA of TransE Entity Embeddings (sample=200)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    # Deduplicate legend entries
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)

    out_path = reports_dir / "tsne_embeddings.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  t-SNE plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load splits
    train_path = KGE_DATA_DIR / "train.txt"
    valid_path = KGE_DATA_DIR / "valid.txt"
    test_path  = KGE_DATA_DIR / "test.txt"

    print("Loading splits …")
    train = _load_split(train_path)
    valid = _load_split(valid_path)
    test  = _load_split(test_path)
    print(f"  train={len(train)}, valid={len(valid)}, test={len(test)}")

    entity2id  = _load_mapping(KGE_DATA_DIR / "entity2id.txt")
    relation2id = _load_mapping(KGE_DATA_DIR / "relation2id.txt")
    n_ent = len(entity2id)
    n_rel = len(relation2id)
    print(f"  Entities={n_ent}, Relations={n_rel}")

    # Limit training triples to 50k for speed
    MAX_TRAIN = 50_000
    train_sample = train[:MAX_TRAIN]
    print(f"  Training sample: {len(train_sample)} triples (max {MAX_TRAIN})")

    all_triples_set = list(set(train + valid + test))

    # --- TransE ---
    print("\n=== TransE ===")
    transe = TransEModel(n_ent, n_rel, dim=50, lr=0.01, n_epochs=100, batch_size=256)
    transe.train(train_sample)

    # --- DistMult ---
    print("\n=== DistMult ===")
    distmult = DistMultModel(n_ent, n_rel, dim=50, lr=0.01, n_epochs=100, batch_size=256)
    distmult.train(train_sample)

    # --- Evaluation ---
    print("\n=== Evaluation ===")
    transe_metrics   = evaluate_model(transe,   test, all_triples_set, model_name="TransE")
    distmult_metrics = evaluate_model(distmult, test, all_triples_set, model_name="DistMult")

    results = {
        "TransE":   transe_metrics,
        "DistMult": distmult_metrics,
    }

    # Save JSON
    metrics_path = REPORTS_DIR / "kge_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # Print table
    header = f"{'Model':<12} {'MRR':>8} {'Hits@1':>8} {'Hits@3':>8} {'Hits@10':>8}"
    print("\n" + header)
    print("-" * len(header))
    for model_name, m in results.items():
        print(f"{model_name:<12} {m['MRR']:>8.4f} {m['Hits@1']:>8.4f} {m['Hits@3']:>8.4f} {m['Hits@10']:>8.4f}")

    # --- t-SNE visualization ---
    print("\n=== t-SNE Visualization ===")
    plot_tsne(transe.ent_emb, entity2id, REPORTS_DIR, n_sample=200)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
