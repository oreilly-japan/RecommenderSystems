from abc import ABC, abstractmethod
from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
from util.models import Dataset, RecommendResult


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass

    def run_sample(self) -> None:
        # Movielens 데이터 취득
        movielens = DataLoader(num_users=1000, num_test_items=5, data_path="../data/ml-10M100K/").load()
        # 추천 계산
        recommend_result = self.recommend(movielens)
        # 추천 결과 평가
        metrics = MetricCalculator().calc(
            movielens.test.rating.tolist(),
            recommend_result.rating.tolist(),
            movielens.test_user2items,
            recommend_result.user2items,
            k=10,
        )
        print(metrics)
