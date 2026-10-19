import numpy as np
import streamlit as st

from app_lib.ui import (DRIVER_COLOR, colhead, driver_legend, esc, figures, load_json, load_recommender, need,
                        preset_users, ticket)
from src.config import K
from src.models import top_k

st.title("Why was this recommended?")
st.html('<p class="lede">The slide asks for recommendations that are "explainable enough". Here is what the '
        'explanation is built from, and an honest test of how much it actually explains.</p>')

rec = load_recommender()
summary = load_json("summary.json")
if rec is None:
    need("the trained model")
    st.stop()

presets = preset_users()
c1, c2 = st.columns([1.2, 2])
who = c1.selectbox("User", list(presets), key="why_user")
user_id = presets[who]
recs = rec.recommend(user_id=user_id)
pick = c2.selectbox("Recommendation", range(len(recs)), format_func=lambda i: f"{i + 1}. {recs[i]['title']}",
                    key="why_item")
r = recs[pick]
col = rec.cat.item_index[r["item_id"]]
liked, seen = rec.rows_for_users([user_id])
cols = liked.indices
contrib = rec.knn.sim[cols, col]
order = np.argsort(-contrib)
total = contrib.sum()

left, right = st.columns([1, 1.15], gap="large")
with left:
    st.html(ticket(pick + 1, r))
    st.html(driver_legend())
    d = r["drivers"]
    st.html("<table class='plain'><tr><th>Part of the hybrid score</th><th>Share</th></tr>" + "".join(
        f"<tr><td><span class='dot' style='background:{DRIVER_COLOR[k]}'></span>{k}</td><td>{v:.0%}</td></tr>"
        for k, v in d.items()) + "</table>")
    st.html('<p class="muted"><b>your movies</b> = item-kNN, <b>taste vectors</b> = ALS, '
            '<b>popularity</b> = how many people liked it, <b>genres</b> = content similarity.</p>')
with right:
    st.html(colhead("Your liked movies that point to it", f"{(contrib > 0).sum()} of {len(cols)} contribute"))
    st.html('<p class="muted">Item-kNN adds up a similarity for every liked movie. The top 3 become the '
            '"because you liked" text, but here they are only a small slice of the total.</p>')
    rows = []
    for c in order[:10]:
        if contrib[c] <= 0:
            break
        w = contrib[c] / contrib[order[0]]
        rows.append(f"<tr><td>{esc(rec.cat.movies.iloc[cols[c]]['clean_title'])}</td>"
                    f"<td><span class='bar' style='width:{w * 140:.0f}px'></span> {contrib[c] / total:.0%}</td></tr>")
    st.html("<table class='plain'><tr><th>Movie</th><th>Share of the item-kNN score</th></tr>" + "".join(rows) + "</table>")

    st.subheader("Deletion test")
    if st.button("Remove the 3 cited movies and re-rank", key="del"):
        L2 = liked.tolil()
        for c in order[:3]:
            L2[0, cols[c]] = 0
        L2 = L2.tocsr()
        L2.eliminate_zeros()
        S = rec.scores("hybrid", L2)
        new_rank = int(np.where(np.argsort(-np.where(seen.toarray()[0] > 0, -np.inf, S[0])) == col)[0][0]) + 1
        if new_rank > K:
            st.success(f"It drops from #{pick + 1} to #{new_rank}: the cited movies really were the reason.")
        else:
            move = f"stays at #{new_rank}" if new_rank == pick + 1 else f"only moves from #{pick + 1} to #{new_rank}"
            st.warning(f"It {move}: other liked movies and popularity keep it in the list, so the "
                       "explanation is partial.")

e = summary.get("explanations", {})
if e:
    st.subheader("Across 3,000 recommendations to 300 random returning users")
    st.html(figures([
        (f"{e['share_citing_a_liked_movie']:.0%}", "cite at least one movie the user liked"),
        (f"{e['mean_knn_score_share_of_top3_citations']:.0%}", "of the item-kNN score comes from the 3 cited movies"),
        (f"{e['share_dropping_out_when_citations_removed']:.0%}", "fall out of the top 10 when those 3 are removed"),
    ]))
    st.html('<div class="callout">"Because you liked X, Y, Z" is true but incomplete. A recommendation is usually '
            'supported by dozens of the user\'s movies plus popularity, so naming three is a readable summary, not '
            'the full reason. That is why each card also shows the score breakdown bar, and the reason text '
            'switches to "Mostly popularity" when popularity is the bigger part.</div>')
