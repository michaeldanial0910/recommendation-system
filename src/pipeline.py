"""One object that holds every fitted model and turns a user (or a few picked movies)
into a ranked, explained top-K list. Used by evaluation, the API and the Streamlit app.
"""
import json

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import K, MODELS, RELEVANT_MIN
from src.models import (ALS, Catalog, ContentBased, DemographicPopularity, ItemKNN, Popularity,
                        interaction_matrix, minmax_rows, top_k)

# Tuned on the validation window (python -m src.tune writes reports/tuning.csv).
DEFAULT_PARAMS = {
    "knn_k": 400, "knn_shrink": 100.0,
    "als_factors": 16, "als_alpha": 1.0, "als_reg": 10.0, "als_iters": 8,
    "w_knn": 1.0, "w_als": 0.25, "w_pop": 0.5, "w_content": 0.1,
}
# New user who picked a few movies at sign-up and gave age + gender: popularity within their
# demographic group replaces global popularity (tuned on validation new users, --stage onboard).
ONBOARD_WEIGHTS = {"knn": 0.5, "als": 0.5, "demo": 1.0}
COMPONENTS = ["knn", "als", "pop", "content", "demo"]


class Recommender:
    def fit(self, history: pd.DataFrame, movies: pd.DataFrame, users_df: pd.DataFrame, params=None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        p = self.params
        self.cat = Catalog(movies)
        self.users_df = users_df
        self.history = history[["user_id", "item_id", "rating", "ts"]]
        self.known_users = sorted(history["user_id"].unique())
        self.uidx = {u: r for r, u in enumerate(self.known_users)}
        # Collaborative models learn from liked movies only: the validation run showed that
        # "who rated what" (including 1-2 star ratings) makes them worse than popularity.
        self.X_liked = interaction_matrix(history, self.known_users, self.cat, liked_only=True)
        self.X_seen = interaction_matrix(history, self.known_users, self.cat)
        self.pop = Popularity().fit(history, self.cat)
        self.demo = DemographicPopularity().fit(history, self.cat, users_df)
        self.knn = ItemKNN(p["knn_k"], p["knn_shrink"]).fit(self.X_liked)
        self.als = ALS(p["als_factors"], p["als_reg"], p["als_alpha"], p["als_iters"]).fit(self.X_liked)
        self.content = ContentBased().fit(self.cat, history)
        n = np.zeros(self.cat.n_items)
        cnt = history["item_id"].map(self.cat.item_index).value_counts()
        n[cnt.index.to_numpy()] = cnt.to_numpy()
        self.n_ratings = n
        liked_n = self.pop.pop
        self.pop_pct = pd.Series(liked_n).rank(pct=True).to_numpy()   # 1.0 = most liked
        return self

    # ---- rows ------------------------------------------------------------------------------
    def rows_for_users(self, user_ids):
        """(liked, seen) sparse rows for users known at fit time; empty rows for unknown users."""
        liked = sp.vstack([self.X_liked[self.uidx[u]] if u in self.uidx else
                           sp.csr_matrix((1, self.cat.n_items)) for u in user_ids]).tocsr()
        seen = sp.vstack([self.X_seen[self.uidx[u]] if u in self.uidx else
                          sp.csr_matrix((1, self.cat.n_items)) for u in user_ids]).tocsr()
        return liked, seen

    def rows_from_items(self, item_lists):
        rows, cols = [], []
        for r, items in enumerate(item_lists):
            for i in items:
                if i in self.cat.item_index:
                    rows.append(r)
                    cols.append(self.cat.item_index[i])
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(item_lists), self.cat.n_items))

    # ---- scores ------------------------------------------------------------------------------
    def component_scores(self, liked_rows) -> dict:
        n = liked_rows.shape[0]
        return {
            "knn": self.knn.score(liked_rows),
            "als": self.als.score(liked_rows),
            "content": self.content.score(liked_rows),
            "pop": np.tile(self.pop.pop, (n, 1)),
        }

    def onboard_scores(self, liked_rows, demo_rows: np.ndarray) -> np.ndarray:
        comps = self.component_scores(liked_rows)
        comps["demo"] = demo_rows
        return self.hybrid(comps, ONBOARD_WEIGHTS)

    def hybrid(self, comps: dict, w: dict | None = None) -> np.ndarray:
        w = w or {c: self.params.get(f"w_{c}", 0) for c in COMPONENTS}
        out = np.zeros_like(comps["pop"], dtype=np.float64)
        for c in COMPONENTS:
            if w.get(c, 0):
                out += w[c] * minmax_rows(comps[c])
        return out

    def scores(self, model: str, liked_rows, user_ids=None) -> np.ndarray:
        if model == "popularity":
            return np.tile(self.pop.pop, (liked_rows.shape[0], 1))
        if model == "demographic_popularity":
            return self.demo.score_users(user_ids)
        if model == "hybrid":
            return self.hybrid(self.component_scores(liked_rows))
        if model == "hybrid_onboard":
            return self.onboard_scores(liked_rows, self.demo.score_users(user_ids))
        return {"item_knn": self.knn, "als": self.als, "content": self.content}[model].score(liked_rows)

    # ---- one user, explained -------------------------------------------------------------------
    def recommend(self, user_id=None, liked_items=None, gender=None, age=None, k=K, model="hybrid"):
        """Top-k for a known user, or for a new user from picked movies and/or demographics.

        Returns a list of dicts: item_id, title, year, genres, score, reason, because (movie ids).
        """
        if user_id is not None and user_id in self.uidx:
            liked, seen = self.rows_for_users([user_id])
        else:
            liked = self.rows_from_items([liked_items or []])
            seen = liked
        n_liked = liked.nnz
        route = model
        if n_liked == 0 and model in ("hybrid", "item_knn", "als", "content"):
            route = "demographic_popularity" if gender and age else "popularity"
        elif model == "hybrid" and user_id not in self.uidx and gender and age:
            route = "hybrid_onboard"
        drivers = None
        if route == "demographic_popularity":
            S = self._demo_scores(gender, age)[None, :]
        elif route in ("hybrid", "hybrid_onboard"):
            comps = self.component_scores(liked)
            if route == "hybrid_onboard":
                comps["demo"] = self._demo_scores(gender, age)[None, :]
                w = ONBOARD_WEIGHTS
            else:
                w = {c: self.params.get(f"w_{c}", 0) for c in COMPONENTS}
            parts = {c: w[c] * minmax_rows(comps[c])[0] for c in COMPONENTS if w.get(c, 0)}
            S = sum(parts.values())[None, :]
            drivers = parts
        else:
            S = self.scores(route, liked)
        idx = top_k(S, seen, k)[0]
        liked_cols = liked.indices
        profile = self.content.profiles(liked)[0] if n_liked else None
        out = []
        for col in idx:
            m = self.cat.movies.iloc[col]
            because, reason = self._explain(route, liked_cols, profile, col, gender, age)
            share = None
            if drivers is not None:
                raw = {"your movies": drivers.get("knn", 0)[col] if "knn" in drivers else 0,
                       "taste vectors": drivers["als"][col] if "als" in drivers else 0,
                       "popularity": (drivers["pop"][col] if "pop" in drivers else 0)
                       + (drivers["demo"][col] if "demo" in drivers else 0),
                       "genres": drivers["content"][col] if "content" in drivers else 0}
                tot = sum(raw.values()) or 1.0
                share = {key: float(v / tot) for key, v in raw.items()}
                cf = share["your movies"] + share["taste vectors"]
                if because and share["popularity"] > cf:
                    group = self._group_text(gender, age) if route == "hybrid_onboard" else "everyone"
                    reason = (f"Mostly popularity ({group}); also liked by fans of "
                              f"{self.cat.movies.iloc[self.cat.item_index[because[0]]]['clean_title']}")
            out.append({"item_id": int(m["item_id"]), "title": m["clean_title"], "year": int(m["year"]),
                        "genres": list(m["genres"]), "score": float(S[0, col]), "reason": reason,
                        "because": because, "model": route, "drivers": share})
        return out

    def _group_text(self, gender, age):
        from src.config import AGE_GROUPS
        g = {"M": "men", "F": "women"}.get(gender, "people")
        return f"{g} aged {AGE_GROUPS.get(age, age)}"

    def similar(self, item_id: int, k=K):
        """Movies most often liked by the same people (item-kNN neighbours)."""
        col = self.cat.item_index[item_id]
        sims = self.knn.sim[:, col].astype(np.float64)
        sims[col] = -np.inf
        idx = np.argsort(-sims)[:k]
        out = []
        for c in idx:
            if sims[c] <= 0:
                break
            m = self.cat.movies.iloc[c]
            out.append({"item_id": int(m["item_id"]), "title": m["clean_title"], "year": int(m["year"]),
                        "genres": list(m["genres"]), "similarity": float(sims[c])})
        return out

    def _demo_scores(self, gender, age):
        return self.demo.group_pop.get((gender, age), self.demo.global_pop)

    def _explain(self, route, liked_cols, profile, col, gender, age):
        title = lambda c: self.cat.movies.iloc[c]["clean_title"]
        n_liked_pop = int(self.pop.pop[col])
        if route == "popularity":
            return [], f"{n_liked_pop:,} people rated it 4+ stars"
        if route == "demographic_popularity":
            return [], f"Popular with {self._group_text(gender, age)}"
        if route == "content":
            gs = self.content.explain(profile, col)
            return [], "Matches the genres you like: " + ", ".join(gs) if gs else "Similar genre mix"
        # item_knn, als, hybrid: cite the liked movies most similar to the recommendation
        cites, _ = self.knn.explain(liked_cols, col) if route != "als" else self.als.explain(liked_cols, col)
        if not cites and route in ("hybrid", "hybrid_onboard"):
            cites, _ = self.als.explain(liked_cols, col)
        ids = [int(self.cat.movies.iloc[c]["item_id"]) for c, _ in cites]
        if cites:
            return ids, "Because you liked " + ", ".join(title(c) for c, _ in cites)
        if self.n_ratings[col] == 0:
            gs = self.content.explain(profile, col)
            return [], "New movie, matches your genres: " + ", ".join(gs)
        return [], f"Popular: {n_liked_pop:,} people rated it 4+"

    # ---- persistence ------------------------------------------------------------------------------
    def save(self, path=None):
        path = path or MODELS / "recommender.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        (MODELS / "params.json").write_text(json.dumps(self.params, indent=2))
        return path

    @staticmethod
    def load(path=None) -> "Recommender":
        return joblib.load(path or MODELS / "recommender.joblib")
