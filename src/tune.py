"""Choose hyper-parameters on the validation window only (train -> val). The test window is
never touched here.

    python -m src.tune --stage cf       # item-kNN and ALS settings, all vs liked-only data
    python -m src.tune --stage hybrid   # blend weights, using the best cf settings
"""
import argparse
import itertools

import pandas as pd

from src.config import K, ONBOARDING_PICKS, PROCESSED, RELEVANT_MIN, REPORTS
from src.metrics import evaluate_lists
from src.models import ALS, Catalog, ItemKNN, Popularity, interaction_matrix, minmax_rows, top_k
from src.pipeline import DEFAULT_PARAMS, Recommender


def setup():
    r = pd.read_parquet(PROCESSED / "ratings.parquet")
    movies = pd.read_parquet(PROCESSED / "movies.parquet")
    users = pd.read_parquet(PROCESSED / "users.parquet")
    hist, val = r[r["split"] == "train"], r[r["split"] == "val"]
    truth = val[val["rating"] >= RELEVANT_MIN].groupby("user_id")["item_id"].apply(set)
    known = set(hist["user_id"])
    warm = [u for u in truth.index if u in known]
    return hist, movies, users, truth, warm


def score_lists(S, seen_rows, cat, warm, truth):
    idx = top_k(S, seen_rows, K)
    recs = {u: cat.ids(idx[j]) for j, u in enumerate(warm)}
    return evaluate_lists(recs, {u: truth[u] for u in warm}, K).mean()


def stage_cf():
    hist, movies, users, truth, warm = setup()
    cat = Catalog(movies)
    known = sorted(hist["user_id"].unique())
    ui = {u: i for i, u in enumerate(known)}
    sel = [ui[u] for u in warm]
    X_all = interaction_matrix(hist, known, cat)
    X_liked = interaction_matrix(hist, known, cat, liked_only=True)
    seen = X_all[sel]
    rows = [{"model": "popularity", "data": "-", "setting": "-",
             **score_lists(Popularity().fit(hist, cat).score_users(warm, None), seen, cat, warm, truth)}]
    for data, X in [("all ratings", X_all), ("liked only", X_liked)]:
        for k, sh in itertools.product([100, 400, 1000], [10, 100]):
            S = ItemKNN(k, sh).fit(X).score(X[sel])
            rows.append({"model": "item_knn", "data": data, "setting": f"k={k} shrink={sh}",
                         **score_lists(S, seen, cat, warm, truth)})
        for f, a in itertools.product([16, 64], [1, 5]):
            S = ALS(f, 10.0, a, 8).fit(X).score(X[sel])
            rows.append({"model": "als", "data": data, "setting": f"factors={f} alpha={a}",
                         **score_lists(S, seen, cat, warm, truth)})
        print("done", data)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "tuning_cf.csv", index=False)
    print(out.sort_values("ndcg", ascending=False).head(10).round(4).to_string(index=False))


def stage_hybrid():
    hist, movies, users, truth, warm = setup()
    rec = Recommender().fit(hist, movies, users)
    liked, seen = rec.rows_for_users(warm)
    comps = rec.component_scores(liked)
    rows = []
    grid = [0, 0.25, 0.5, 1.0]
    for wk, wa, wp, wc in itertools.product(grid, grid, [0, 0.25, 0.5], [0, 0.1]):
        if wk == wa == wp == wc == 0:
            continue
        S = rec.hybrid(comps, {"knn": wk, "als": wa, "pop": wp, "content": wc})
        rows.append({"w_knn": wk, "w_als": wa, "w_pop": wp, "w_content": wc,
                     **score_lists(S, seen, rec.cat, warm, truth)})
    out = pd.DataFrame(rows).sort_values("ndcg", ascending=False)
    out.to_csv(REPORTS / "tuning_hybrid.csv", index=False)
    print(out.head(10).round(4).to_string(index=False))
    print("current default:", {k: v for k, v in DEFAULT_PARAMS.items() if k.startswith("w_")})


def stage_onboard():
    """New users in the validation window: first 3 liked movies are their sign-up picks."""
    hist, movies, users, _, _ = setup()
    val = pd.read_parquet(PROCESSED / "ratings.parquet").query("split == 'val'")
    rec = Recommender().fit(hist, movies, users)
    liked = val[val["rating"] >= RELEVANT_MIN].sort_values("ts")
    picks, rest = {}, {}
    for u, g in liked[~liked["user_id"].isin(set(hist["user_id"]))].groupby("user_id"):
        items = g["item_id"].tolist()
        if len(items) > ONBOARDING_PICKS:
            picks[u], rest[u] = items[:ONBOARDING_PICKS], set(items[ONBOARDING_PICKS:])
    us = list(picks)
    L = rec.rows_from_items([picks[u] for u in us])
    comps = rec.component_scores(L)
    comps["demo"] = rec.demo.score_users(us)
    rows = []
    for wk, wa, wd in itertools.product([0, 0.25, 0.5, 1], [0, 0.25, 0.5], [0, 0.25, 0.5, 1, 2]):
        if wk == wa == wd == 0:
            continue
        S = rec.hybrid(comps, {"knn": wk, "als": wa, "demo": wd})
        idx = top_k(S, L, K)
        recs = {u: rec.cat.ids(idx[j]) for j, u in enumerate(us)}
        rows.append({"w_knn": wk, "w_als": wa, "w_demo": wd, **evaluate_lists(recs, rest, K).mean()})
    out = pd.DataFrame(rows).sort_values("ndcg", ascending=False)
    out.to_csv(REPORTS / "tuning_onboard.csv", index=False)
    print(len(us), "validation new users")
    print(out.head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cf", "hybrid", "onboard"], required=True)
    {"cf": stage_cf, "hybrid": stage_hybrid, "onboard": stage_onboard}[ap.parse_args().stage]()
