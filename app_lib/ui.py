"""Shared look, cached data and small HTML components for the Streamlit app."""
import html
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import AGE_GROUPS, MODELS, OCCUPATIONS, PROCESSED, RELEVANT_MIN, REPORTS

# ---- design tokens (same family as the churn / fraud / sentiment apps) --------------------
INK = "#1C2333"
INK_SOFT = "#5A6478"
RULE = "#E3E6EC"
PAPER = "#F7F8FA"
HIT = "#13976A"
MODEL_COLOR = {"popularity": "#8A94A6", "demographic_popularity": "#B07D00", "content": "#D4562E",
               "item_knn": "#2A6FD6", "als": "#6250D6", "hybrid": "#13976A", "hybrid_onboard": "#0B5E42"}
MODEL_NAME = {"popularity": "Popularity", "demographic_popularity": "Popular in age/gender group",
              "content": "Content (genres)", "item_knn": "Item-kNN", "als": "ALS",
              "hybrid": "Hybrid", "hybrid_onboard": "Hybrid + demographics"}
# Score drivers inside the hybrid (categorical, validated for colour-blind contrast).
DRIVER_COLOR = {"your movies": "#2A6FD6", "taste vectors": "#6250D6", "popularity": "#8A94A6", "genres": "#D4562E"}
SEGMENT_NAME = {"warm": "Returning users", "new_cold": "New users, nothing rated",
                "new_onboard": "New users after 3 picks"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
html, body, .stApp, p, li, label, h1, h2, h3, h4, td, th, input, select,
button p, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ max-width: 1180px; padding-top: 4.2rem; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 600; font-size: 2.3rem; line-height: 1.15; }}
h2 {{ font-weight: 600; font-size: 1.4rem; margin-top: 1.6rem; }}
h3 {{ font-weight: 600; font-size: 1.08rem; }}
p, li {{ line-height: 1.6; }}
.lede {{ color: {INK_SOFT}; font-size: 1.06rem; max-width: 72ch; margin-top: -0.4rem; }}
.muted {{ color: {INK_SOFT}; font-size: 0.9rem; }}
.colhead {{ display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
            border-bottom: 2px solid {INK}; padding-bottom: 6px; margin-bottom: 8px; }}
.colhead .t {{ font-weight: 600; font-size: 1.02rem; }}
.colhead .s {{ font-size: 0.85rem; color: {INK_SOFT}; font-variant-numeric: tabular-nums; }}

/* a movie "ticket": rank, title, year, genres, reason, score drivers */
.ticket {{ display: grid; grid-template-columns: 34px 1fr; gap: 10px; background: #fff; border: 1px solid {RULE};
           border-radius: 10px; padding: 11px 14px; margin: 0 0 8px; position: relative; }}
.ticket.hit {{ border-color: {HIT}; box-shadow: inset 4px 0 0 {HIT}; }}
.ticket .rank {{ font-size: 1.25rem; font-weight: 600; color: {INK_SOFT}; font-variant-numeric: tabular-nums; line-height: 1.3; }}
.ticket .title {{ font-weight: 600; font-size: 1.0rem; line-height: 1.3; }}
.ticket .title small {{ font-weight: 400; color: {INK_SOFT}; margin-left: 6px; }}
.ticket .reason {{ font-size: 0.86rem; color: {INK_SOFT}; margin-top: 3px; line-height: 1.45; }}
.g.hitchip {{ background: {HIT}; color: #fff; font-weight: 600; }}
.genres {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }}
.g {{ font-size: 0.72rem; padding: 1px 7px; border-radius: 999px; background: #EEF0F4; color: {INK_SOFT}; }}
.drivers {{ display: flex; height: 6px; border-radius: 3px; overflow: hidden; gap: 1px; margin-top: 8px; background: #fff; }}
.drivers span {{ display: block; height: 100%; }}
.legend {{ display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.8rem; color: {INK_SOFT}; margin: 2px 0 10px; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 5px; vertical-align: 0; }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 3px 11px;
         font-size: 0.84rem; font-weight: 500; border: 1px solid {RULE}; background: #fff; color: {INK}; }}
.route {{ display: inline-block; font-size: 0.82rem; font-weight: 600; padding: 4px 12px; border-radius: 999px;
          color: #fff; margin-bottom: 10px; }}
.person {{ background: #fff; border: 1px solid {RULE}; border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; }}
.person .who {{ font-weight: 600; }}
.person .meta {{ font-size: 0.86rem; color: {INK_SOFT}; }}
.watched {{ font-size: 0.9rem; line-height: 1.55; margin: 0; padding-left: 1.1rem; }}
.watched li {{ margin: 0; }}
.stars {{ color: #B07D00; letter-spacing: 1px; font-size: 0.8rem; }}

.figure-row {{ display: flex; gap: 28px; flex-wrap: wrap; margin: 10px 0 4px; }}
.figure {{ min-width: 150px; }}
.figure .n {{ font-size: 2rem; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.figure .l {{ font-size: 0.86rem; color: {INK_SOFT}; max-width: 24ch; }}
.callout {{ border-left: 4px solid {INK}; background: #fff; padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 12px 0; }}
.callout.warn {{ border-left-color: #B07D00; }}
table.plain {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 0.93rem; }}
table.plain th {{ text-align: left; font-weight: 500; color: {INK_SOFT}; border-bottom: 1px solid {RULE}; padding: 7px 10px; }}
table.plain td {{ border-bottom: 1px solid {RULE}; padding: 7px 10px; }}
table.plain td.best {{ font-weight: 600; }}
.bar {{ height: 8px; border-radius: 4px; background: #2A6FD6; display: inline-block; vertical-align: middle; }}
@media (max-width: 640px) {{ h1 {{ font-size: 1.8rem; }} .figure .n {{ font-size: 1.6rem; }} }}
</style>
"""


def setup():
    st.html(CSS)


def esc(s) -> str:
    return html.escape(str(s))


def dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


def stars(r: int) -> str:
    return f'<span class="stars">{"★" * int(r)}{"☆" * (5 - int(r))}</span>'


def driver_legend() -> str:
    return '<div class="legend">' + "".join(f"<span>{dot(c)}{k}</span>" for k, c in DRIVER_COLOR.items()) + "</div>"


def ticket(rank: int, rec: dict, hit: bool = False, show_drivers=True) -> str:
    genres = "".join(f'<span class="g">{esc(g)}</span>' for g in rec["genres"])
    strip = ""
    if show_drivers and rec.get("drivers"):
        segs = "".join(f'<span style="width:{max(v, 0.004) * 100:.1f}%;background:{DRIVER_COLOR[k]}" '
                       f'title="{k} {v:.0%}"></span>' for k, v in rec["drivers"].items())
        strip = f'<div class="drivers">{segs}</div>'
    if hit:
        genres += '<span class="g hitchip">✓ rated 4+ later</span>'
    return (f'<div class="ticket{" hit" if hit else ""}"><div class="rank">{rank}</div><div>'
            f'<div class="title">{esc(rec["title"])}<small>{rec["year"]}</small></div>'
            f'<div class="genres">{genres}</div><div class="reason">{esc(rec["reason"])}</div>{strip}</div></div>')


def colhead(title: str, sub: str = "", color: str | None = None) -> str:
    d = dot(color) if color else ""
    return f'<div class="colhead"><span class="t">{d}{esc(title)}</span><span class="s">{esc(sub)}</span></div>'


def figures(items) -> str:
    cells = "".join(f'<div class="figure"><div class="n">{esc(n)}</div><div class="l">{esc(l)}</div></div>'
                    for n, l in items)
    return f'<div class="figure-row">{cells}</div>'


def person_card(u: pd.Series, n_liked: int, n_rated: int) -> str:
    g = {"M": "Man", "F": "Woman"}[u["gender"]]
    return (f'<div class="person"><div class="who">User {int(u["user_id"])}</div>'
            f'<div class="meta">{g}, {AGE_GROUPS[int(u["age"])]}, {OCCUPATIONS[int(u["occupation"])]}<br>'
            f'{n_rated} ratings before 2 Dec 2000, {n_liked} of them 4+ stars</div></div>')


# ---- data ----------------------------------------------------------------------------------
@st.cache_resource
def load_recommender():
    from src.pipeline import Recommender
    path = MODELS / "recommender.joblib"
    return Recommender.load(path) if path.exists() else None


@st.cache_data
def load_ratings() -> pd.DataFrame | None:
    path = PROCESSED / "ratings.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def load_users() -> pd.DataFrame | None:
    path = PROCESSED / "users.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def demo_users() -> pd.DataFrame:
    """Returning test users worth showing: real history and at least 5 liked movies later."""
    r = load_ratings()
    hist, test = r[r["split"] != "test"], r[r["split"] == "test"]
    h = hist.groupby("user_id").agg(n_rated=("rating", "size"),
                                    n_liked=("rating", lambda x: int((x >= RELEVANT_MIN).sum())))
    t = test[test["rating"] >= RELEVANT_MIN].groupby("user_id").size().rename("n_later")
    d = h.join(t, how="inner")
    d = d[(d["n_liked"].between(15, 120)) & (d["n_later"] >= 5)]
    return d.join(load_users().set_index("user_id")).reset_index()



@st.cache_data
def preset_users() -> dict:
    """Pick one demo user per taste profile by their most-liked genre (deterministic)."""
    r = load_ratings()
    rec = load_recommender()
    d = demo_users()
    hist = r[(r["split"] != "test") & (r["rating"] >= RELEVANT_MIN) & r["user_id"].isin(d["user_id"])]
    g = hist.merge(rec.cat.movies[["item_id", "genres"]], on="item_id").explode("genres")
    share = g.groupby(["user_id", "genres"]).size().unstack(fill_value=0)
    share = share.div(share.sum(axis=1), axis=0)
    out = {}
    for label, cols in [("Sci-fi & action fan", ["Sci-Fi", "Action"]), ("Drama & classics", ["Drama", "Film-Noir"]),
                        ("Comedy & family", ["Comedy", "Children's", "Animation"]),
                        ("Thriller & crime", ["Thriller", "Crime"])]:
        score = share[cols].sum(axis=1).drop(list(out.values()), errors="ignore")
        out[label] = int(score.idxmax())
    return out


def need(what: str):
    st.info(f"This view needs {what}. Run `python -m src.data` then `python -m src.evaluate` once "
            "(see README, *How to run*).")


def altair_theme():
    font = "IBM Plex Sans, system-ui, -apple-system, Helvetica, Arial, sans-serif"
    return {"config": {
        "font": font, "background": "transparent", "view": {"stroke": None},
        "axis": {"labelColor": INK_SOFT, "titleColor": INK_SOFT, "gridColor": RULE, "domainColor": RULE,
                 "tickColor": RULE, "labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 500},
        "legend": {"labelColor": INK, "titleColor": INK_SOFT, "labelFontSize": 12, "titleFontWeight": 500},
        "header": {"labelColor": INK, "labelFontSize": 13, "labelFontWeight": 500, "titleColor": INK_SOFT},
        "title": {"color": INK, "fontWeight": 600, "fontSize": 14, "anchor": "start"}}}


def use_altair_theme():
    alt.theme.register("plex", enable=True)(altair_theme)
