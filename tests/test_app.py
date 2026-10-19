"""Smoke tests: every page renders, and the interactive bits change what is shown.

    pytest -q tests/test_app.py
"""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["../app_pages/for_you.py", "../app_pages/new_user.py", "../app_pages/results.py",
         "../app_pages/why_this.py", "../app_pages/data_story.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(page, default_timeout=120).run()
    assert not at.exception, at.exception


def test_hindsight_toggle_marks_hits():
    at = AppTest.from_file("../app_pages/for_you.py", default_timeout=120).run()
    before = "".join(h.proto.body for h in at.get("html"))
    at.toggle[0].set_value(True).run()
    after = "".join(h.proto.body for h in at.get("html"))
    assert "rated 4+ later" in after and "rated 4+ later" not in before


def test_new_user_route_changes():
    at = AppTest.from_file("../app_pages/new_user.py", default_timeout=120).run()
    route = lambda: next(h.proto.body for h in at.get("html") if "Route:" in h.proto.body)
    assert "Route: Popularity" in route()
    at.selectbox(key="nu_age").set_value(25).run()
    at.segmented_control(key="nu_gender").set_value("F").run()
    assert "age/gender" in route()
    at.multiselect(key="nu_picks").set_value([2571, 1196, 260]).run()
    assert "Hybrid + demographics" in route()
