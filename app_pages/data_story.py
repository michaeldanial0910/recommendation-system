import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.ui import INK, figures, load_ratings, load_recommender, need, use_altair_theme
from src.config import RELEVANT_MIN, REPORTS

use_altair_theme()
st.title("The data: MovieLens 1M, split by time")
st.html('<p class="lede">One million star ratings that 6,040 people gave 3,706 movies on MovieLens between '
        'April 2000 and February 2003, plus each user\'s gender, age group and occupation and each movie\'s '
        'genres and year. Official GroupLens file, checksum verified.</p>')

r = load_ratings()
rec = load_recommender()
if r is None or rec is None:
    need("the processed data")
    st.stop()


@st.cache_data
def weekly(r):
    w = pd.to_datetime(r["ts"], unit="s").dt.to_period("W").dt.start_time
    return r.assign(week=w).groupby(["week", "split"]).size().rename("ratings").reset_index()


n_u, n_i = r["user_id"].nunique(), r["item_id"].nunique()
st.html(figures([(f"{len(r):,}", "ratings"), (f"{n_u:,}", "users, each with 20+ ratings"),
                 (f"{1 - len(r) / (n_u * n_i):.1%}", "of the user × movie grid is empty"),
                 (f"{(r['rating'] >= RELEVANT_MIN).mean():.0%}", "of ratings are 4 or 5 stars (our 'liked')")]))

st.subheader("Why a time split, not a random one")
w = weekly(r)
colors = alt.Scale(domain=["train", "val", "test"], range=["#2A6FD6", "#B07D00", "#13976A"])
chart = alt.Chart(w).mark_bar(width=3).encode(
    x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=8)), y=alt.Y("ratings:Q", stack=None, scale=alt.Scale(type="log", domainMin=50), title="Ratings per week (log)"),
    color=alt.Color("split:N", scale=colors, legend=alt.Legend(orient="top", title=None)),
    tooltip=[alt.Tooltip("week:T"), "split:N", alt.Tooltip("ratings:Q", format=",")])
st.altair_chart(chart.properties(height=260), theme=None, width="stretch")
st.html('<div class="callout">A random split would let the model learn from ratings made <i>after</i> the ones it '
        'is tested on. It would know which movies became popular later and what other users went on to watch, '
        'which no live system can know. Here every model sees only ratings before <b>22 Nov 2000</b> while tuning '
        '(validated on 22 Nov to 2 Dec) and only ratings before <b>2 Dec 2000</b> for the final test. That is '
        'the same position as a recommender deployed on that date.</div>')
s = r.groupby("split").agg(ratings=("rating", "size"), users=("user_id", "nunique"))
hist_users = set(r[r["split"] != "test"]["user_id"])
test_users = set(r[r["split"] == "test"]["user_id"])
new_items = set(r[r["split"] == "test"]["item_id"]) - set(r[r["split"] != "test"]["item_id"])
st.html(figures([(f"{s.loc['train', 'ratings']:,}", "train ratings (to 22 Nov 2000)"),
                 (f"{s.loc['val', 'ratings']:,}", "validation ratings, used only for tuning"),
                 (f"{s.loc['test', 'ratings']:,}", "test ratings (2 Dec 2000 → Feb 2003)"),
                 (f"{len(test_users - hist_users)}", "test users never seen before: real cold start"),
                 (f"{len(new_items)}", "movies first rated in the test period")]))

st.subheader("A long tail: why popularity is a strong baseline")
liked = r[(r["split"] != "test") & (r["rating"] >= RELEVANT_MIN)]
counts = liked["item_id"].value_counts().to_numpy()
cum = np.cumsum(counts) / counts.sum()
top = int(np.searchsorted(cum, 0.5)) + 1
curve = pd.DataFrame({"movies": np.arange(1, len(cum) + 1) / len(cum), "likes": cum})
lc = alt.Chart(curve).mark_area(color="#8A94A6", opacity=0.5, line={"color": INK}).encode(
    x=alt.X("movies:Q", axis=alt.Axis(format="%"), title="Movies, most liked first"),
    y=alt.Y("likes:Q", axis=alt.Axis(format="%"), title="Share of all 4+ ratings"))
st.altair_chart(lc.properties(height=240), theme=None, width="stretch")
st.html(f'<p class="muted">The {top} most-liked movies ({top / len(counts):.0%} of rated titles) collect half of all '
        f'4- and 5-star ratings. A list of blockbusters is therefore right for a lot of people, and any '
        f'personalised model has to beat that, not a random list.</p>')

st.subheader("Two choices that matter")
tune = pd.read_csv(REPORTS / "tuning_cf.csv") if (REPORTS / "tuning_cf.csv").exists() else None
if tune is not None:
    best = tune.groupby(["model", "data"])["ndcg"].max()
    vals = {k: f"{best[k]:.3f}" for k in best.index}
    pop = vals[("popularity", "-")]
    sentence = (f"Trained on every rating, item-kNN and ALS scored <i>below</i> popularity on validation "
                f"({vals[('item_knn', 'all ratings')]} / {vals[('als', 'all ratings')]} vs {pop}). Trained only on "
                f"4–5 star ratings they beat it ({vals[('item_knn', 'liked only')]} / {vals[('als', 'liked only')]}).")
else:
    sentence = "See reports/tuning_cf.csv."
st.html('<table class="plain"><tr><th>Choice</th><th>Why</th></tr>'
        '<tr><td>A hit = a later rating of 4 or 5</td><td>A recommender should suggest movies people will enjoy, '
        'not just ones they will watch. A 1-star rating is evidence of a bad recommendation.</td></tr>'
        f'<tr><td>Collaborative models learn from liked movies only</td><td>{sentence}</td></tr></table>')
