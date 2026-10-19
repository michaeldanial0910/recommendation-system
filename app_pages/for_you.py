import streamlit as st

from app_lib.ui import (MODEL_COLOR, MODEL_NAME, colhead, demo_users, driver_legend, esc, figures, load_csv, load_ratings,
                        load_recommender, need, person_card, preset_users, stars, ticket)
from src.config import RELEVANT_MIN

st.title("Ten movies for someone we already know")
st.html('<p class="lede">The model is frozen on 2 December 2000. Pick a real MovieLens user, see what they had '
        'liked by then, and compare the hybrid recommender with the most-popular list. Then switch on hindsight '
        'to see which recommendations they actually went on to rate 4 or 5 stars.</p>')

rec, ratings = load_recommender(), load_ratings()
if rec is None or ratings is None:
    need("the trained model")
    st.stop()

presets = preset_users()
pool = demo_users()
c1, c2, c3 = st.columns([4.2, 1, 1], vertical_alignment="bottom")
choice = c1.pills("Taste profile", list(presets) + ["Random user"], default=list(presets)[0], key="persona")
if choice == "Random user":
    if "random_user" not in st.session_state or c2.button("Another one", width="stretch"):
        st.session_state.random_user = int(pool["user_id"].sample(1).iloc[0])
    user_id = st.session_state.random_user
else:
    user_id = presets[choice or list(presets)[0]]
hindsight = c3.toggle("Hindsight", value=False, help="Mark the movies this user rated 4+ after the cutoff.")

hist = ratings[(ratings["user_id"] == user_id) & (ratings["split"] != "test")]
later = set(ratings[(ratings["user_id"] == user_id) & (ratings["split"] == "test")
                    & (ratings["rating"] >= RELEVANT_MIN)]["item_id"])
liked = hist[hist["rating"] >= RELEVANT_MIN].sort_values(["rating", "ts"], ascending=[False, False])
u = pool.set_index("user_id").loc[user_id]
u["user_id"] = user_id

hyb = rec.recommend(user_id=user_id, model="hybrid")
pop = rec.recommend(user_id=user_id, model="popularity")
hits_h = sum(r["item_id"] in later for r in hyb)
hits_p = sum(r["item_id"] in later for r in pop)

left, mid, right = st.columns([1, 1.25, 1.25], gap="medium")
with left:
    st.html(person_card(u, len(liked), len(hist)))
    st.html(colhead("What they liked", "before the cutoff"))
    titles = rec.cat.movies.set_index("item_id")
    items = "".join(f'<li>{esc(titles.loc[i, "clean_title"])} {stars(r)}</li>'
                    for i, r in zip(liked["item_id"].head(14), liked["rating"].head(14)))
    more = f'<p class="muted">… and {len(liked) - 14} more</p>' if len(liked) > 14 else ""
    st.html(f'<ul class="watched">{items}</ul>{more}')
with mid:
    sub = f"{hits_h}/10 rated 4+ later" if hindsight else "personalised"
    st.html(colhead("Hybrid recommender", sub, MODEL_COLOR["hybrid"]))
    st.html(driver_legend())
    st.html("".join(ticket(i + 1, r, hindsight and r["item_id"] in later) for i, r in enumerate(hyb)))
with right:
    sub = f"{hits_p}/10 rated 4+ later" if hindsight else "same for everyone"
    st.html(colhead("Popularity baseline", sub, MODEL_COLOR["popularity"]))
    st.html(driver_legend().replace('class="legend"', 'class="legend" style="visibility:hidden"'))
    st.html("".join(ticket(i + 1, r, hindsight and r["item_id"] in later, show_drivers=False)
                    for i, r in enumerate(pop)))

if hindsight:
    allh = load_csv("warm_user_hits.csv")
    items = [(f"{hits_h} vs {hits_p}", "hits in the top 10 for this user: hybrid vs popularity")]
    if allh is not None:
        win = (allh["hybrid_hits"] > allh["popularity_hits"]).mean()
        tie = (allh["hybrid_hits"] == allh["popularity_hits"]).mean()
        items += [(f"{win:.0%}", "of all 1,122 returning test users get more hits from the hybrid"),
                  (f"{tie:.0%} / {1 - win - tie:.0%}", "tie / get fewer hits than with popularity"),
                  (f"{allh['hybrid_hits'].mean():.2f} vs {allh['popularity_hits'].mean():.2f}", "average hits per user")]
    st.html(figures(items))
    st.html('<p class="muted">One user is an anecdote. The Results page has the averages with confidence '
            'intervals. Movies they had already rated are never recommended again.</p>')
