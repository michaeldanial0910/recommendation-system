import streamlit as st

from app_lib.ui import MODEL_COLOR, MODEL_NAME, driver_legend, esc, load_csv, load_recommender, need, ticket
from src.config import AGE_GROUPS, ONBOARDING_PICKS

st.title("A brand-new user: the cold-start problem")
st.html('<p class="lede">A new user has rated nothing, so collaborative filtering has nothing to work from. '
        'The recommender changes route as it learns more: global popularity, then popularity within the '
        'user\'s age and gender group, then a hybrid seeded by the movies they pick at sign-up.</p>')

rec = load_recommender()
if rec is None:
    need("the trained model")
    st.stop()

movies = rec.cat.movies
rated = movies[rec.n_ratings > 0].assign(n=rec.n_ratings[rec.n_ratings > 0]).sort_values("n", ascending=False)
label = {int(r.item_id): f"{r.clean_title} ({r.year})" for r in rated.itertuples()}

c1, c2, c3 = st.columns([1, 1.2, 3], vertical_alignment="top")
gender = c1.segmented_control("Gender", ["F", "M"], format_func={"F": "Woman", "M": "Man"}.get, key="nu_gender")
age = c2.selectbox("Age group", [None] + list(AGE_GROUPS), format_func=lambda a: "Not given" if a is None
                   else AGE_GROUPS[a], key="nu_age")
picks = c3.multiselect(f"Pick up to {ONBOARDING_PICKS} movies you liked", list(label), format_func=label.get,
                       max_selections=ONBOARDING_PICKS, key="nu_picks",
                       placeholder="Search a title, e.g. Toy Story, Alien, Casablanca")

recs = rec.recommend(liked_items=picks, gender=gender, age=age)
route = recs[0]["model"]
EXPLAIN = {
    "popularity": "Nothing known yet: the most-liked movies overall.",
    "demographic_popularity": "Only age and gender known: the most-liked movies in that group.",
    "hybrid": "Picks but no age/gender: similar movies to the picks, blended with overall popularity.",
    "hybrid_onboard": "Picks plus age and gender: similar movies to the picks, blended with the group's favourites.",
}

left, right = st.columns([1.3, 1], gap="large")
with left:
    st.html(f'<span class="route" style="background:{MODEL_COLOR[route]}">Route: {esc(MODEL_NAME[route])}</span>'
            f'<p class="muted" style="margin-top:-4px">{EXPLAIN[route]}</p>')
    if route.startswith("hybrid"):
        st.html(driver_legend())
    st.html("".join(ticket(i + 1, r) for i, r in enumerate(recs)))
with right:
    st.subheader("How each route scored on real new users")
    met = load_csv("metrics.csv")
    v = (lambda seg, mod: f"{met.set_index(['segment', 'model']).loc[(seg, mod), 'ndcg']:.3f}") if met is not None \
        else (lambda seg, mod: "n/a")
    st.html(f"""<table class="plain">
      <tr><th>Route</th><th>What it knows</th><th>NDCG@10</th></tr>
      <tr><td>Popularity</td><td>nothing</td><td>{v('new_cold', 'popularity')}</td></tr>
      <tr><td>Popular in age/gender group</td><td>age, gender</td><td class="best">{v('new_cold', 'demographic_popularity')}</td></tr>
      <tr><td>Hybrid</td><td>3 picks</td><td>{v('new_onboard', 'hybrid')}</td></tr>
      <tr><td>Hybrid + demographics</td><td>3 picks, age, gender</td><td class="best">{v('new_onboard', 'hybrid_onboard')}</td></tr>
    </table>""")
    st.html('<p class="muted">640 users who joined after the cutoff. First two rows: their whole first test '
            'period. Last two: the first 3 movies they liked are treated as sign-up picks and the rest are '
            'scored, which is a harder target, so compare rows within a pair, not across.</p>')
    pairs = load_csv("paired_vs_popularity.csv")
    d = None
    if pairs is not None:
        q = pairs[(pairs["segment"] == "new_onboard") & (pairs["model"] == "hybrid_onboard")
                  & (pairs["vs"] == "demographic_popularity") & (pairs["metric"] == "ndcg")]
        d = q.iloc[0] if len(q) else None
    gap = (f"{d['diff']:+.3f} (95% CI {d['lo']:+.3f} to {d['hi']:+.3f})" if d is not None else "small")
    verdict = "the interval includes zero" if d is None or d["lo"] <= 0 <= d["hi"] else "a real but small gain"
    st.html(f'<div class="callout warn"><b>What did not work.</b> With only 3 picks, item-kNN and ALS on their own '
            f'score <i>below</i> plain popularity ({v("new_onboard", "item_knn")} and {v("new_onboard", "als")} vs '
            f'{v("new_onboard", "popularity")}). Blending the picks with the age/gender favourites beats '
            f'global popularity, but its lead over demographics alone is {gap}: {verdict}. '
            f'Three picks add little beyond age and gender; personalisation pays off once a user has rated '
            f'about 10 movies (Results, "Where personalisation helps").</div>')
