"""Ranking metrics for top-K lists, computed per user so they can be bootstrapped.

Every function takes one user's ranked list and that user's set of relevant items.
"""
import numpy as np
import pandas as pd


def per_user_metrics(ranked: list[int], relevant: set[int], k: int) -> dict:
    top = ranked[:k]
    hits = np.array([i in relevant for i in top], dtype=float)
    n_rel = len(relevant)
    n_hit = hits.sum()
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float((hits * discounts).sum())
    idcg = float(discounts[: min(n_rel, k)].sum())
    # Average precision, normalised by the best achievable number of hits in K slots.
    prec_at_hit = np.cumsum(hits) / np.arange(1, len(top) + 1)
    ap = float((prec_at_hit * hits).sum() / min(n_rel, k)) if n_rel else 0.0
    return {
        "precision": n_hit / k,
        "recall": n_hit / n_rel if n_rel else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "map": ap,
        "hit_rate": float(n_hit > 0),
    }


def evaluate_lists(recs: dict[int, list[int]], truth: dict[int, set[int]], k: int) -> pd.DataFrame:
    """One row per user (only users with at least one relevant item)."""
    rows = []
    for u, rel in truth.items():
        if not rel:
            continue
        m = per_user_metrics(recs.get(u, []), rel, k)
        m["user_id"] = u
        rows.append(m)
    return pd.DataFrame(rows).set_index("user_id")


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """Mean of (a - b) over the same users with a bootstrap 95% interval."""
    return bootstrap_ci(a - b, n_boot, seed)


def coverage(recs: dict[int, list[int]], catalog_size: int, k: int) -> float:
    items = {i for lst in recs.values() for i in lst[:k]}
    return len(items) / catalog_size


def novelty(recs: dict[int, list[int]], pop_rank_pct: dict[int, float], k: int) -> float:
    """Mean popularity percentile of recommended items (1.0 = the most-rated movie)."""
    vals = [pop_rank_pct.get(i, 0.0) for lst in recs.values() for i in lst[:k]]
    return float(np.mean(vals)) if vals else float("nan")
