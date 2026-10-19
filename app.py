"""Streamlit app: movie recommender with cold-start handling (MovieLens 1M).

    streamlit run app.py        ->  http://localhost:8501
"""
import streamlit as st

st.set_page_config(page_title="Movie recommender", page_icon="🎬", layout="wide")

from app_lib.ui import setup  # noqa: E402  (set_page_config must run first)

setup()
pages = [
    st.Page("app_pages/for_you.py", title="For you", url_path="for-you", default=True),
    st.Page("app_pages/new_user.py", title="New user", url_path="new-user"),
    st.Page("app_pages/results.py", title="Results", url_path="results"),
    st.Page("app_pages/why_this.py", title="Why this?", url_path="why"),
    st.Page("app_pages/data_story.py", title="The data", url_path="data"),
]
st.navigation(pages, position="top").run()
