import numpy as np
import pytest

from src.metrics import per_user_metrics


def test_perfect_list():
    m = per_user_metrics([1, 2, 3], {1, 2, 3}, k=3)
    assert m["precision"] == 1 and m["recall"] == 1 and m["ndcg"] == pytest.approx(1) and m["map"] == 1


def test_no_hits():
    m = per_user_metrics([4, 5, 6], {1}, k=3)
    assert m["hit_rate"] == 0 and m["ndcg"] == 0 and m["map"] == 0


def test_rank_matters():
    first = per_user_metrics([1, 9, 9], {1}, k=3)["ndcg"]
    last = per_user_metrics([9, 9, 1], {1}, k=3)["ndcg"]
    assert first == pytest.approx(1) and last == pytest.approx(1 / np.log2(4))


def test_map_hand_computed():
    # hits at ranks 1 and 3, 2 relevant: AP = (1/1 + 2/3) / 2
    assert per_user_metrics([1, 7, 2], {1, 2}, k=3)["map"] == pytest.approx((1 + 2 / 3) / 2)
