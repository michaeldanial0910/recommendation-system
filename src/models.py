"""Recommenders. Every model has fit(history) and score(user_rows) -> (n_users, n_items) array.

The item axis is the full MovieLens catalogue (3,883 movies), so a movie nobody had rated
before the cutoff still has a column: collaborative models give it no score, the content
model can.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import RELEVANT_MIN, SEED


@dataclass
class Catalog:
    movies: pd.DataFrame                       # one row per movie, index position = item index
    item_index: dict = field(init=False)

    def __post_init__(self):
        self.movies = self.movies.reset_index(drop=True)
        self.item_index = {m: i for i, m in enumerate(self.movies["item_id"])}

    @property
    def n_items(self) -> int:
        return len(self.movies)

    def ids(self, idx) -> list[int]:
        return self.movies["item_id"].to_numpy()[np.asarray(idx)].tolist()


def interaction_matrix(history: pd.DataFrame, users: list[int], cat: Catalog,
                       liked_only=False) -> sp.csr_matrix:
    """Binary users x items matrix ("has rated", or "rated 4+" when liked_only)."""
    h = history[history["rating"] >= RELEVANT_MIN] if liked_only else history
    uidx = {u: r for r, u in enumerate(users)}
    h = h[h["user_id"].isin(uidx)]
    rows = h["user_id"].map(uidx).to_numpy()
    cols = h["item_id"].map(cat.item_index).to_numpy()
    return sp.csr_matrix((np.ones(len(h)), (rows, cols)), shape=(len(users), cat.n_items))


# ---- baselines -------------------------------------------------------------------------
class Popularity:
    """Most movies rated 4+ in the history window: the same list for everyone."""
    name = "popularity"

    def fit(self, history, cat: Catalog, users_df=None):
        liked = history[history["rating"] >= RELEVANT_MIN]
        counts = liked["item_id"].map(cat.item_index).value_counts()
        self.pop = np.zeros(cat.n_items)
        self.pop[counts.index.to_numpy()] = counts.to_numpy()
        return self

    def score_users(self, user_ids, X):
        return np.tile(self.pop, (len(user_ids), 1))


class DemographicPopularity:
    """Most liked movies among users of the same gender and age group (needs no ratings)."""
    name = "demographic_popularity"

    def fit(self, history, cat: Catalog, users_df: pd.DataFrame):
        self.users = users_df.set_index("user_id")
        liked = history[history["rating"] >= RELEVANT_MIN].merge(users_df, on="user_id")
        liked["col"] = liked["item_id"].map(cat.item_index)
        self.global_pop = np.bincount(liked["col"], minlength=cat.n_items).astype(float)
        self.group_pop = {}
        for key, g in liked.groupby(["gender", "age"]):
            v = np.bincount(g["col"], minlength=cat.n_items).astype(float)
            # shrink small groups towards the global ranking
            self.group_pop[key] = v / v.sum() + 0.1 * self.global_pop / self.global_pop.sum()
        return self

    def score_users(self, user_ids, X=None):
        out = np.empty((len(user_ids), len(self.global_pop)))
        for r, u in enumerate(user_ids):
            row = self.users.loc[u]
            out[r] = self.group_pop.get((row["gender"], row["age"]), self.global_pop)
        return out


# ---- collaborative filtering -------------------------------------------------------------
class ItemKNN:
    """Item-item cosine similarity on who-rated-what, keeping each item's top neighbours.

    score(u, j) = sum over the user's movies h of sim(h, j). The terms of that sum are the
    explanation: "recommended because you rated h1, h2, h3".
    """
    name = "item_knn"

    def __init__(self, k_neighbors=100, shrink=10.0):
        self.k, self.shrink = k_neighbors, shrink

    def fit(self, X: sp.csr_matrix):
        Xc = X.tocsc().astype(np.float64)
        co = (Xc.T @ Xc).toarray()
        n = np.diag(co).copy()
        sim = co / (np.sqrt(np.outer(n, n)) + self.shrink)
        np.fill_diagonal(sim, 0.0)
        if self.k and self.k < sim.shape[0]:
            # keep the k largest similarities per target column
            thresh = -np.partition(-sim, self.k - 1, axis=0)[self.k - 1]
            sim[sim < thresh[None, :]] = 0.0
        self.sim = sim.astype(np.float32)
        return self

    def score(self, X_rows: sp.csr_matrix) -> np.ndarray:
        return np.asarray(X_rows @ self.sim)

    def explain(self, history_cols: np.ndarray, item_col: int, top=3):
        contrib = self.sim[history_cols, item_col]
        order = np.argsort(-contrib)[:top]
        return [(int(history_cols[i]), float(contrib[i])) for i in order if contrib[i] > 0], float(contrib.sum())


class ALS:
    """Implicit-feedback matrix factorisation (Hu, Koren & Volinsky 2008), solved exactly.

    Each user and movie gets a vector of `factors` numbers; a rated movie counts as a
    preference with confidence 1 + alpha.
    """
    name = "als"

    def __init__(self, factors=64, reg=0.1, alpha=10.0, iters=10, seed=SEED):
        self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def _solve(self, M: sp.csr_matrix, Y: np.ndarray) -> np.ndarray:
        YtY = Y.T @ Y
        out = np.zeros((M.shape[0], self.f))
        eye = self.reg * np.eye(self.f)
        for r in range(M.shape[0]):
            idx = M.indices[M.indptr[r]:M.indptr[r + 1]]
            if len(idx) == 0:
                continue
            Yi = Y[idx]
            A = YtY + self.alpha * (Yi.T @ Yi) + eye
            b = (1 + self.alpha) * Yi.sum(axis=0)
            out[r] = np.linalg.solve(A, b)
        return out

    def fit(self, X: sp.csr_matrix):
        rng = np.random.default_rng(self.seed)
        n_u, n_i = X.shape
        self.U = rng.normal(0, 0.01, (n_u, self.f))
        self.V = rng.normal(0, 0.01, (n_i, self.f))
        Xt = X.T.tocsr()
        for _ in range(self.iters):
            self.U = self._solve(X, self.V)
            self.V = self._solve(Xt, self.U)
        return self

    def fold_in(self, X_rows: sp.csr_matrix) -> np.ndarray:
        """User vectors for any rows (including users not seen in fit) from item vectors."""
        return self._solve(X_rows.tocsr(), self.V)

    def score(self, X_rows: sp.csr_matrix) -> np.ndarray:
        return self.fold_in(X_rows) @ self.V.T

    def explain(self, history_cols: np.ndarray, item_col: int, top=3):
        """Liked movies whose vectors point the same way as the recommended one."""
        V = self.V / (np.linalg.norm(self.V, axis=1, keepdims=True) + 1e-9)
        sims = V[history_cols] @ V[item_col]
        order = np.argsort(-sims)[:top]
        return [(int(history_cols[i]), float(sims[i])) for i in order], None


# ---- content-based ---------------------------------------------------------------------------
class ContentBased:
    """Genres + release decade as features; user profile = average of liked movies.

    Works for movies nobody has rated yet (item cold start) and for a new user who picks a
    few movies (user cold start).
    """
    name = "content"

    def __init__(self, decade_weight=0.5):
        self.decade_weight = decade_weight

    def fit(self, cat: Catalog, history: pd.DataFrame):
        genres = sorted({g for gs in cat.movies["genres"] for g in gs})
        G = np.array([[g in gs for g in genres] for gs in cat.movies["genres"]], dtype=float)
        # idf: a rare genre (Film-Noir) says more than a common one (Drama)
        G *= np.log(len(G) / (1 + G.sum(axis=0)))[None, :]
        decade = (cat.movies["year"] // 10 * 10).clip(lower=1930)
        decades = sorted(decade.unique())
        D = (decade.to_numpy()[:, None] == np.array(decades)[None, :]).astype(float) * self.decade_weight
        F = np.hstack([G, D])
        self.F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        self.feature_names = genres + [f"{d}s" for d in decades]
        self.genres = genres
        return self

    def profiles(self, X_rows: sp.csr_matrix) -> np.ndarray:
        P = np.asarray(X_rows @ self.F)
        return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)

    def score(self, X_rows: sp.csr_matrix) -> np.ndarray:
        return self.profiles(X_rows) @ self.F.T

    def explain(self, profile: np.ndarray, item_col: int, top=3):
        contrib = profile[: len(self.genres)] * self.F[item_col, : len(self.genres)]
        order = np.argsort(-contrib)[:top]
        return [self.genres[i] for i in order if contrib[i] > 0]


def minmax_rows(S: np.ndarray) -> np.ndarray:
    lo, hi = S.min(axis=1, keepdims=True), S.max(axis=1, keepdims=True)
    return (S - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def top_k(scores: np.ndarray, exclude: sp.csr_matrix | None, k: int) -> np.ndarray:
    """Indices of the k best items per row, never recommending something already rated."""
    S = scores.astype(np.float64, copy=True)
    if exclude is not None:
        S[exclude.nonzero()] = -np.inf
    part = np.argpartition(-S, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(S, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)
