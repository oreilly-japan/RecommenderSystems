from abc import ABC, abstractmethod
from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
from util.models import Dataset, RecommendResult


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass

    def run_sample(self) -> None:
        # Movielensのデータを取得
        movielens = DataLoader(num_users=1000, num_test_items=5, data_path="../data/ml-10M100K/").load()
        # 推薦計算
        recommend_result = self.recommend(movielens)
        # 推薦結果の評価
        metrics = MetricCalculator().calc(
            movielens.test.rating.tolist(),
            recommend_result.rating.tolist(),
            movielens.test_user2items,
            recommend_result.user2items,
            k=10,
        )
        print(metrics)
