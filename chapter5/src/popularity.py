from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np

np.random.seed(0)


class PopularityRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 評価数の閾値
        minimum_num_rating = kwargs.get("minimum_num_rating", 200)

        # 各アイテムごとの平均の評価値を計算し、その平均評価値を予測値として利用する
        movie_rating_average = dataset.train.groupby("movie_id").agg({"rating": np.mean})
        # テストデータに予測値を格納する。テストデータのみに存在するアイテムの予測評価値は０とする
        movie_rating_predict = dataset.test.merge(
            movie_rating_average, on="movie_id", how="left", suffixes=("_test", "_pred")
        ).fillna(0)

        # 各ユーザに対するおすすめ映画は、そのユーザがまだ評価していない映画の中から評価値が高いもの10作品とする
        # ただし、評価件数が少ないとノイズが大きいため、minimum_num_rating件以上評価がある映画に絞る
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
