from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np

np.random.seed(0)


class PopularityRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 평갓값의 임곗값
        minimum_num_rating = kwargs.get("minimum_num_rating", 200)

        # 각 아이템별 평균 평갓값을 계산하고, 그 평균 평갓값을 예측값으로 사용한다
        movie_rating_average = dataset.train.groupby("movie_id").agg({"rating": np.mean})
        # 테스트 데이터에 예측값을 저장한다. 테스트 데이터에만 존재하는 아이템의 예측 평갓값은 0으로 한다
        movie_rating_predict = dataset.test.merge(
            movie_rating_average, on="movie_id", how="left", suffixes=("_test", "_pred")
        ).fillna(0)

        # 각 사용자에 대한 추천 영화는 해당 사용자가 아직 평가하지 않은 영화 중에서 평균값이 높은 10개 작품으로 한다
        # 단, 평가 건수가 적으면 노이즈가 커지므로 minimum_num_rating건 이상 평가가 있는 영화로 한정한다
        pred_user2items = defaultdict(list)
        user_watched_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        movie_stats = dataset.train.groupby("movie_id").agg({"rating": [np.size, np.mean]})
        atleast_flg = movie_stats["rating"]["size"] >= minimum_num_rating
        movies_sorted_by_rating = (
            movie_stats[atleast_flg].sort_values(by=("rating", "mean"), ascending=False).index.tolist()
        )

        for user_id in dataset.train.user_id.unique():
            for movie_id in movies_sorted_by_rating:
                if movie_id not in user_watched_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    PopularityRecommender().run_sample()
