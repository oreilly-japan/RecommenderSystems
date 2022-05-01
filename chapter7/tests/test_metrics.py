import pytest
from src.metrics import Metrics

pytest.error_threshold = 0.00001

pytest.true_rating = [0.0, 1.0, 2.0, 3.0, 4.0]
pytest.pred_rating = [0.1, 1.1, 2.1, 3.1, 4.1]


def test_mae():
    assert (
        pytest.approx(
            Metrics().mae(pytest.true_rating, pytest.pred_rating),
            pytest.error_threshold,
        )
        == 0.1
    )


def test_mse():
    assert (
        pytest.approx(
            Metrics().mse(pytest.true_rating, pytest.pred_rating),
            pytest.error_threshold,
        )
        == 0.01
    )


def test_rmse():
    assert (
        pytest.approx(
            Metrics().rmse(pytest.true_rating, pytest.pred_rating),
            pytest.error_threshold,
        )
        == 0.1
    )


pytest.pred_item = [1, 2, 3, 4, 5]
pytest.true_item = [2, 4, 6, 8]


def test_precision_at_k():
    assert Metrics().precision_at_k(pytest.true_item, pytest.pred_item, 5) == 0.4


def test_recall_at_k():
    assert Metrics().recall_at_k(pytest.true_item, pytest.pred_item, 5) == 0.5


def test_f1_at_k():
    assert (
        pytest.approx(
            Metrics().f1_at_k(pytest.true_item, pytest.pred_item, 5),
            pytest.error_threshold,
        )
        == 0.4444444
    )


pytest.user1_relevance = [1, 0, 0]
pytest.user2_relevance = [0, 1, 0]
pytest.user3_relevance = [1, 0, 1]
pytest.users_relevance = [
    pytest.user1_relevance,
    pytest.user2_relevance,
    pytest.user3_relevance,
]


def test_rr_at_k():
    assert Metrics().rr_at_k(pytest.user1_relevance, 3) == 1.0 / 1
    assert Metrics().rr_at_k(pytest.user2_relevance, 3) == 1.0 / 2
    assert Metrics().rr_at_k(pytest.user3_relevance, 3) == 1.0 / 1


def test_mrr_at_k():
    assert Metrics().mrr_at_k(pytest.users_relevance, 3) == (1.0 / 1 + 1.0 / 2 + 1.0 / 1) / 3


def test_ap_at_k():
    assert Metrics().ap_at_k(pytest.user1_relevance, 3) == 1.0 / 1
    assert Metrics().ap_at_k(pytest.user2_relevance, 3) == 1.0 / 2
    assert Metrics().ap_at_k(pytest.user3_relevance, 3) == (1.0 / 1 + 2.0 / 3) / 2


def test_map_at_k():
    assert Metrics().map_at_k(pytest.users_relevance, 3) == (1.0 / 1 + 1.0 / 2 + (1.0 / 1 + 2.0 / 3) / 2) / 3


pytest.user4_weighted_relevance = [0, 2, 0, 1, 0]


def test_dcg_at_k():
    # dcg = 0 + 2 / math.log2(2) + 0 / math.log2(3)
    # + 1 / math.log2(4) + 0 / math.log2(5) = 2.5
    assert Metrics().dcg_at_k(pytest.user4_weighted_relevance, 5) == 2.5


def test_ndcg_at_k():
    # dcg_ideal = 2 + 1 / math.log2(2) + 0 / math.log2(3)
    # + 0 / math.log2(4) + 0 / math.log2(5) = 3.0
    # ndcg = dcg / dcg_ideal = 2.5 / 3.0
    assert Metrics().ndcg_at_k(pytest.user4_weighted_relevance, 5) == 2.5 / 3.0
