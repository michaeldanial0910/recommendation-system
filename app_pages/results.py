import altair as alt
import pandas as pd
import streamlit as st

from app_lib.ui import (INK, MODEL_COLOR, MODEL_NAME, SEGMENT_NAME, figures, load_csv, load_json, need,
                        use_altair_theme)

use_altair_theme()
st.title("Does it beat 'just show the popular movies'?")
st.html('<p class="lede">Every model was fitted on ratings up to 2 December 2000 and scored once on what users '
        'liked afterwards (rated 4 or 5). The main metric is NDCG@10: it rewards putting the movies a user '
        'later liked near the top of a 10-movie list. Intervals are 95% bootstrap over users.</p>')

m, pairs, hist = load_csv("metrics.csv"), load_csv("paired_vs_popularity.csv"), load_csv("by_history.csv")
summary = load_json("summary.json")
if m is None:
    need("reports/metrics.csv")
    st.stop()

g = m.set_index(["segment", "model"])
p = pairs[(pairs["metric"] == "ndcg")].set_index(["segment", "model", "vs"])
d_w = p.loc[("warm", "hybrid", "popularity")]
st.html(figures([
    (f"{g.loc[('warm', 'hybrid'), 'ndcg']:.3f}", "hybrid NDCG@10 for returning users"),
    (f"+{d_w['diff']:.3f}", f"over popularity (paired 95% CI +{d_w['lo']:.3f} to +{d_w['hi']:.3f})"),
    (f"{g.loc[('warm', 'hybrid'), 'hit_rate']:.0%}", "of returning users get at least one later-liked movie in their 10"),
    (f"{summary['item_cold_start']['content_auc']:.2f}", "AUC ranking brand-new movies by genre: no better than chance (0.5)"),
]))

st.subheader("NDCG@10 by user group")
seg = st.segmented_control("Group", list(SEGMENT_NAME), default="warm", format_func=SEGMENT_NAME.get, key="seg")
d = m[m["segment"] == (seg or "warm")].copy()
d["Model"] = d["model"].map(MODEL_NAME)
order = d.sort_values("ndcg", ascending=False)["Model"].tolist()
color = alt.Color("Model:N", scale=alt.Scale(domain=[MODEL_NAME[k] for k in d["model"]],
                                             range=[MODEL_COLOR[k] for k in d["model"]]), legend=None)
tip = [alt.Tooltip("Model:N"), alt.Tooltip("ndcg:Q", format=".3f", title="NDCG@10"),
       alt.Tooltip("ndcg_lo:Q", format=".3f", title="95% low"), alt.Tooltip("ndcg_hi:Q", format=".3f", title="95% high"),
       alt.Tooltip("precision:Q", format=".3f", title="precision@10"), alt.Tooltip("recall:Q", format=".3f", title="recall@10"),
       alt.Tooltip("n_users:Q", title="users")]
base = alt.Chart(d).encode(y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13, labelLimit=260)),
                           color=color, tooltip=tip)
bars = base.mark_bar(height=22, cornerRadiusEnd=3).encode(x=alt.X("ndcg:Q", title="NDCG@10", scale=alt.Scale(domain=[0, 0.5]), axis=alt.Axis(tickCount=5, format=".1f")))
err = base.mark_rule(color=INK, strokeWidth=1.5).encode(x="ndcg_lo:Q", x2="ndcg_hi:Q")
txt = base.mark_text(align="left", dx=6, color=INK, fontSize=12).encode(x="ndcg_hi:Q", text=alt.Text("ndcg:Q", format=".3f"))
st.altair_chart((bars + err + txt).properties(height=46 * len(d)), theme=None, width="stretch")

st.subheader("All ranking metrics")
cols = {"ndcg": "NDCG", "precision": "Precision", "recall": "Recall", "map": "MAP", "hit_rate": "Hit rate",
        "coverage": "Catalogue coverage", "popularity_pct": "Avg. popularity pct."}
t = d.set_index("Model")[list(cols)].rename(columns=cols).sort_values("NDCG", ascending=False)
st.dataframe(t.style.format("{:.3f}").highlight_max(subset=list(cols.values())[:5], props="font-weight:600"),
             width="stretch")
st.html('<p class="muted"><b>Coverage</b> = share of the 3,883 movies that appear in anyone\'s top 10. '
        '<b>Avg. popularity percentile</b> near 1.0 means the lists are mostly blockbusters. Every model except '
        f'content-based has a strong popularity bias, and the hybrid only covers '
        f'{g.loc[("warm", "hybrid"), "coverage"]:.1%} of the catalogue.</p>')

st.subheader("Where personalisation helps")
h = hist[hist["model"].isin(["popularity", "item_knn", "als", "hybrid"])].copy()
h["Model"] = h["model"].map(MODEL_NAME)
order_b = ["0-9", "10-29", "30-99", "100+"]
line = alt.Chart(h).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2.5).encode(
    x=alt.X("liked_before_cutoff:N", sort=order_b, title="Movies the user liked before the cutoff"),
    y=alt.Y("mean:Q", title="NDCG@10", scale=alt.Scale(zero=False)),
    color=alt.Color("Model:N", scale=alt.Scale(domain=[MODEL_NAME[k] for k in ["popularity", "item_knn", "als", "hybrid"]],
                                                 range=[MODEL_COLOR[k] for k in ["popularity", "item_knn", "als", "hybrid"]]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=["Model:N", "liked_before_cutoff:N", alt.Tooltip("mean:Q", format=".3f"), alt.Tooltip("count:Q", title="users")])
st.altair_chart(line.properties(height=300), theme=None, width="stretch")
st.html('<p class="muted">With fewer than 10 liked movies the hybrid is no better than popularity. From 10 liked '
        'movies on, the collaborative models pull ahead and stay ahead.</p>')

st.subheader("Is the difference real? Paired comparison against the baseline")
pp = pairs[pairs["metric"] == "ndcg"].copy()
pp["Group"] = pp["segment"].map(SEGMENT_NAME)
pp["Model"] = pp["model"].map(MODEL_NAME)
pp["Baseline"] = pp["vs"].map(MODEL_NAME)
pp["Verdict"] = pp.apply(lambda r: "better" if r["lo"] > 0 else ("worse" if r["hi"] < 0 else "no clear difference"), axis=1)
pp = pp[pp["model"] != "content"]
st.dataframe(pp[["Group", "Model", "Baseline", "diff", "lo", "hi", "Verdict"]].rename(
    columns={"diff": "NDCG difference", "lo": "95% low", "hi": "95% high"}).style.format(
    {"NDCG difference": "{:+.3f}", "95% low": "{:+.3f}", "95% high": "{:+.3f}"}), hide_index=True, width="stretch")

ic = summary["item_cold_start"]
st.subheader("Brand-new movies (item cold start)")
st.html(f'<div class="callout warn">{ic["new_movies_in_catalog_with_test_ratings"]} movies got their first ratings '
        f'after the cutoff. Collaborative models cannot score them at all. The content model (genres + decade) can, '
        f'but when it ranks those new movies for the {ic["users_who_liked_a_new_movie"]} users who liked one, its '
        f'AUC is <b>{ic["content_auc"]:.2f}</b> (95% CI {ic["auc_lo"]:.2f}–{ic["auc_hi"]:.2f}): indistinguishable '
        f'from random. Genres are not enough. A real fix needs richer metadata (plot, cast, embeddings) or an '
        f'exploration slot that shows new titles to some users to collect first ratings.</div>')
