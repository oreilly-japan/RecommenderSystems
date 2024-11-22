import numpy as np
from sklearn.metrics import mean_squared_error
from util.models import Metrics
from typing import Dict, List


class MetricCalculator:
    def calc(
        self,
        true_rating: List[float],
        pred_rating: List[float],
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> Metrics:
        rmse = self._calc_rmse(true_rating, pred_rating)
        precision_at_k = self._calc_precision_at_k(true_user2items, pred_user2items, k)
        recall_at_k = self._calc_recall_at_k(true_user2items, pred_user2items, k)
        return Metrics(rmse, precision_at_k, recall_at_k)

    def _precision_at_k(self, true_items: List[int], pred_items: List[int], k: int) -> float:
        if k == 0:
            return 0.0

        p_at_k = (len(set(true_items) & set(pred_items[:k]))) / k
        return p_at_k

    def _recall_at_k(self, true_items: List[int], pred_items: List[int], k: int) -> float:
        if len(true_items) == 0 or k == 0:
            return 0.0

        r_at_k = (len(set(true_items) & set(pred_items[:k]))) / len(true_items)
        return r_at_k

    def _calc_rmse(self, true_rating: List[float], pred_rating: List[float]) -> float:
        return np.sqrt(mean_squared_error(true_rating, pred_rating))

    def _calc_recall_at_k(
        self, true_user2items: Dict[int, List[int]], pred_user2items: Dict[int, List[int]], k: int
    ) -> float:
        scores = []
        # テストデータに存在する各ユーザーのrecall@kを計算
        for user_id in true_user2items.keys():
            r_at_k = self._recall_at_k(true_user2items[user_id], pred_user2items[user_id], k)
            scores.append(r_at_k)
        return np.mean(scores)

    def _calc_precision_at_k(
        self, true_user2items: Dict[int, List[int]], pred_user2items: Dict[int, List[int]], k: int
    ) -> float:
        scores = []
        # 테스트 데이터에 존재하는 각 사용자의 precision@k를 계산한다
        for user_id in true_user2items.keys():
            p_at_k = self._precision_at_k(true_user2items[user_id], pred_user2items[user_id], k)
            scores.append(p_at_k)
        return np.mean(scores)
