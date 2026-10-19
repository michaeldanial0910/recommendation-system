"""Fit every model on train+validation (data up to 2 Dec 2000) and evaluate once on test.

    python -m src.evaluate   ->  models/recommender.joblib, reports/*.csv, reports/summary.json

Four test scenarios:
  warm        users with ratings before the cutoff; recommend from their history
  new_cold    users who appear only after the cutoff, nothing known but age and gender
  new_onboard the same new users after they "pick" their first 3 liked movies
  new_items   movies nobody had rated before the cutoff (item cold start)
"""
import json

import numpy as np
import pandas as pd

from src.config import K, ONBOARDING_PICKS, PROCESSED, RELEVANT_MIN, REPORTS, SEED
from src.metrics import bootstrap_ci, coverage, evaluate_lists, novelty, paired_diff_ci
from src.models import top_k
from src.pipeline import Recommender

METRICS = ["precision", "recall", "ndcg", "map", "hit_rate"]
WARM_MODELS = ["popularity", "demographic_popularity", "content", "item_knn", "als", "hybrid"]


def load():
    r = pd.read_parquet(PROCESSED / "ratings.parquet")
    movies = pd.read_parquet(PROCESSED / "movies.parquet")
    users = pd.read_parquet(PROCESSED / "users.parquet")
    return r[r["split"] != "test"], r[r["split"] == "test"], movies, users


def run_models(rec, models, user_ids, liked, seen, truth):
    per_user, lists = {}, {}
    for m in models:
        S = rec.scores(m, liked, user_ids)
        idx = top_k(S, seen, K)
        recs = {u: rec.cat.ids(idx[j]) for j, u in enumerate(user_ids)}
        lists[m] = recs
        per_user[m] = evaluate_lists(recs, truth, K)
    return per_user, lists


def summarise(segment, per_user, lists, rec):
    rows = []
    for m, df in per_user.items():
        row = {"segment": segment, "model": m, "n_users": len(df)}
        for met in METRICS:
            mean, lo, hi = bootstrap_ci(df[met].to_numpy(), seed=SEED)
            row.update({met: mean, f"{met}_lo": lo, f"{met}_hi": hi})
        pct = dict(zip(rec.cat.movies["item_id"], rec.pop_pct))
        row["coverage"] = coverage(lists[m], rec.cat.n_items, K)
        row["popularity_pct"] = novelty(lists[m], pct, K)
        rows.append(row)
    return rows


def paired(segment, per_user, base="popularity"):
    rows = []
    for m, df in per_user.items():
        if m == base:
            continue
        b = per_user[base].loc[df.index]
        for met in ["ndcg", "precision", "recall"]:
            mean, lo, hi = paired_diff_ci(df[met].to_numpy(), b[met].to_numpy(), seed=SEED)
            rows.append({"segment": segment, "model": m, "vs": base, "metric": met,
                         "diff": mean, "lo": lo, "hi": hi})
    return rows


def explanation_checks(rec, warm, liked, seen, n_users=300):
    """Does the 'because you liked X' text actually drive the recommendation?

    Deletion test: remove the cited movies from the user's history and re-rank. If the
    recommendation falls out of the top 10, the cited movies were really why it was there.
    """
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(warm), size=min(n_users, len(warm)), replace=False)
    total = cited = dropped = 0
    share = []
    for j in pick:
        L, Sn = liked[j], seen[j]
        if L.nnz == 0:
            continue
        S = rec.scores("hybrid", L)
        top = top_k(S, Sn, K)[0]
        for col in top:
            total += 1
            cites, knn_total = rec.knn.explain(L.indices, col)
            if not cites:
                continue
            cited += 1
            share.append(sum(c for _, c in cites) / knn_total if knn_total else 0)
            L2 = L.copy().tolil()
            for c, _ in cites:
                L2[0, c] = 0
            L2 = L2.tocsr()
            L2.eliminate_zeros()
            if col not in top_k(rec.scores("hybrid", L2), Sn, K)[0]:
                dropped += 1
    return {"recommendations_checked": total,
            "share_citing_a_liked_movie": cited / total,
            "mean_knn_score_share_of_top3_citations": float(np.mean(share)),
            "share_dropping_out_when_citations_removed": dropped / cited}


def item_cold_start(rec, hist, test, warm_ids, liked_rows):
    """Rank only the movies nobody had rated before the cutoff, by content score.
    AUC = chance that a new movie the user liked ranks above one they did not like or rate."""
    new_cols = np.where(rec.n_ratings == 0)[0]
    new_ids = set(rec.cat.ids(new_cols))
    t = test[test["item_id"].isin(new_ids) & (test["rating"] >= RELEVANT_MIN)]
    liked_new = t.groupby("user_id")["item_id"].apply(set)
    pos = {u: j for j, u in enumerate(warm_ids)}
    S = rec.content.score(liked_rows)
    aucs = []
    for u, items in liked_new.items():
        if u not in pos or liked_rows[pos[u]].nnz == 0:
            continue
        s = S[pos[u], new_cols]
        is_pos = np.isin(rec.cat.ids(new_cols), list(items))
        if is_pos.all() or not is_pos.any():
            continue
        ranks = pd.Series(s).rank().to_numpy()
        n_p, n_n = is_pos.sum(), (~is_pos).sum()
        aucs.append((ranks[is_pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))
    aucs = np.array(aucs)
    mean, lo, hi = bootstrap_ci(aucs, seed=SEED)
    return {"new_movies_in_catalog_with_test_ratings": int(test[test["item_id"].isin(new_ids)]["item_id"].nunique()),
            "users_who_liked_a_new_movie": int(len(aucs)), "content_auc": mean, "auc_lo": lo, "auc_hi": hi,
            "random_auc": 0.5, "collaborative_models": "cannot score them (no ratings yet)"}


def main():
    hist, test, movies, users = load()
    rec = Recommender().fit(hist, movies, users)
    rec.save()
    liked_test = test[test["rating"] >= RELEVANT_MIN].sort_values("ts")
    known = set(hist["user_id"])
    summary_rows, paired_rows = [], []

    # ---- warm users ----
    truth = liked_test.groupby("user_id")["item_id"].apply(set)
    warm = [u for u in truth.index if u in known]
    liked, seen = rec.rows_for_users(warm)
    pu, lists = run_models(rec, WARM_MODELS, warm, liked, seen, {u: truth[u] for u in warm})
    summary_rows += summarise("warm", pu, lists, rec)
    # per-user hits in the top 10 (hybrid vs popularity), for the app's hindsight view
    hits = pd.DataFrame({"hybrid_hits": (pu["hybrid"]["precision"] * K).round().astype(int),
                         "popularity_hits": (pu["popularity"]["precision"] * K).round().astype(int)})
    hits.to_csv(REPORTS / "warm_user_hits.csv")
    paired_rows += paired("warm", pu)
    # by history size
    n_hist = pd.Series(np.asarray(liked.sum(axis=1)).ravel(), index=warm)
    bins = [-1, 9, 29, 99, 10_000]
    labels = ["0-9", "10-29", "30-99", "100+"]
    bucket = pd.cut(n_hist, bins=bins, labels=labels)
    by_hist = []
    for m in ["popularity", "item_knn", "als", "hybrid"]:
        df = pu[m].join(bucket.rename("liked_before_cutoff"))
        g = df.groupby("liked_before_cutoff", observed=True)["ndcg"].agg(["mean", "count"]).reset_index()
        g["model"] = m
        by_hist.append(g)
    pd.concat(by_hist).to_csv(REPORTS / "by_history.csv", index=False)

    # ---- new users: nothing known but demographics ----
    new_users = [u for u in truth.index if u not in known]
    pu_c, lists_c = run_models(rec, ["popularity", "demographic_popularity"], new_users,
                               *rec.rows_for_users(new_users), {u: truth[u] for u in new_users})
    summary_rows += summarise("new_cold", pu_c, lists_c, rec)
    paired_rows += paired("new_cold", pu_c)

    # ---- new users after onboarding: first 3 liked movies become their "picks" ----
    picks, rest = {}, {}
    for u, g in liked_test[liked_test["user_id"].isin(new_users)].groupby("user_id"):
        items = g["item_id"].tolist()
        if len(items) > ONBOARDING_PICKS:
            picks[u], rest[u] = items[:ONBOARDING_PICKS], set(items[ONBOARDING_PICKS:])
    ob_users = list(picks)
    L = rec.rows_from_items([picks[u] for u in ob_users])
    pu_o, lists_o = run_models(rec, WARM_MODELS + ["hybrid_onboard"], ob_users, L, L, rest)
    summary_rows += summarise("new_onboard", pu_o, lists_o, rec)
    paired_rows += paired("new_onboard", pu_o)
    paired_rows += [dict(r, vs="demographic_popularity") for r in paired("new_onboard", pu_o, base="demographic_popularity")]

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORTS / "metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(REPORTS / "paired_vs_popularity.csv", index=False)

    extra = {
        "item_cold_start": item_cold_start(rec, hist, test, warm, liked),
        "explanations": explanation_checks(rec, warm, liked, seen),
        "users": {"warm": len(warm), "new_cold": len(new_users), "new_onboard": len(ob_users)},
        "params": rec.params,
    }
    (REPORTS / "summary.json").write_text(json.dumps(extra, indent=2))
    cols = ["segment", "model", "n_users", "ndcg", "ndcg_lo", "ndcg_hi", "precision", "recall", "map",
            "hit_rate", "coverage", "popularity_pct"]
    print(summary[cols].round(3).to_string(index=False))
    print(json.dumps({k: extra[k] for k in ["item_cold_start", "explanations", "users"]}, indent=2))


if __name__ == "__main__":
    main()
