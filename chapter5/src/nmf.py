from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
from sklearn.decomposition import NMF

np.random.seed(0)


class NMFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 결손값을 채우는 방법
        fillna_with_zero = kwargs.get("fillna_with_zero", True)
        factors = kwargs.get("factors", 5)

        # 평갓값을 사용자 x 영화의 행렬로 변환한다. 결손값은 평균값 또는 0으로 채운다)
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

        # 예측 평갓값 행렬
        pred_matrix = np.dot(P, Q)

        # 학습용에 나오지 않은 사용자나 영화의 예측 평갓값은 평균 평갓값으로 한다
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

        # 각 사용자에게 대한 추천 영화는 그 사용자가 아직 평가하지 않은 영화 중에서 예측값이 높은 순으로 한다
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
