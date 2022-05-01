from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
from sklearn.decomposition import NMF

np.random.seed(0)


class NMFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 欠損値の穴埋め方法
        fillna_with_zero = kwargs.get("fillna_with_zero", True)
        factors = kwargs.get("factors", 5)

        # 評価値をユーザー×映画の行列に変換。欠損値は、平均値または０で穴埋めする
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")
        user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
        movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))
        if fillna_with_zero:
            matrix = user_movie_matrix.fillna(0).to_numpy()
        else:
            matrix = user_movie_matrix.fillna(dataset.train.rating.mean()).to_numpy()

        nmf = NMF(n_components=factors)
        nmf.fit(matrix)
        P = nmf.fit_transform(matrix)
        Q = nmf.components_

        # 予測評価値行列
        pred_matrix = np.dot(P, Q)

        # 学習用に出てこないユーザーや映画の予測評価値は、平均評価値とする
        average_score = dataset.train.rating.mean()
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            if user_id not in user_id2index or row["movie_id"] not in movie_id2index:
                pred_results.append(average_score)
                continue
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        # 各ユーザに対するおすすめ映画は、そのユーザがまだ評価していない映画の中から予測値が高い順にする
        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in dataset.train.user_id.unique():
            if user_id not in user_id2index:
                continue
            user_index = user_id2index[row["user_id"]]
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = user_movie_matrix.columns[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    NMFRecommender().run_sample()
