from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
import implicit
from scipy.sparse import lil_matrix

np.random.seed(0)


class BPRRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 因子数
        factors = kwargs.get("factors", 10)
        # 評価数の閾値
        minimum_num_rating = kwargs.get("minimum_num_rating", 0)
        # エポック数
        n_epochs = kwargs.get("n_epochs", 50)

        # 行列分解用に行列を作成する
        filtered_movielens_train = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= minimum_num_rating
        )

        movielens_train_high_rating = filtered_movielens_train[dataset.train.rating >= 4]

        unique_user_ids = sorted(movielens_train_high_rating.user_id.unique())
        unique_movie_ids = sorted(movielens_train_high_rating.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        movielens_matrix = lil_matrix((len(unique_movie_ids), len(unique_user_ids)))
        for i, row in movielens_train_high_rating.iterrows():
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            movielens_matrix[movie_index, user_index] = 1.0

        # initialize a model
        model = implicit.bpr.BayesianPersonalizedRanking(factors=factors, iterations=n_epochs)

        # 学習
        model.fit(movielens_matrix)

        # 推薦
        recommendations = model.recommend_all(movielens_matrix.T)
        pred_user2items = defaultdict(list)
        for user_id, user_index in user_id2index.items():
            movie_indexes = recommendations[user_index, :]
            for movie_index in movie_indexes:
                movie_id = unique_movie_ids[movie_index]
                pred_user2items[user_id].append(movie_id)
        # BPRでは評価値の予測は難しいため、rmseの評価は行わない。（便宜上、テストデータの予測値をそのまま返す）
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    BPRRecommender().run_sample()
