from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np

np.random.seed(0)


class RandomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 사용자 ID와 아이템 ID에 대해 0부터 시작하는 인덱스를 할당한다
        unique_user_ids = sorted(dataset.train.user_id.unique())
        unique_movie_ids = sorted(dataset.train.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        # 사용자 x 아이템의 행렬에서 각 셀의 예측 평갓값은 0.5~5.0의 균등 난수로 한다
        pred_matrix = np.random.uniform(0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids)))

        # rmse 평가용으로 테스트 데이터에 나오는 사용자와 아이템의 예측 평갓값을 저장한다
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            # 테스트 데이터의 아이템 ID가 학습용으로 등장하지 않는 경우도 난수를 저장한다
            if row["movie_id"] not in movie_id2index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue
            # 테스트 데이터에 나타나는 사용자 ID와 아이템 ID의 인덱스를 얻어, 평갓값 행렬값을 얻는다
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        # 순위 평가용 데이터 작성
        # 각 사용자에 대한 추천 영화는, 해당 사용자가 아직 평가하지 않은 영화 중에서 무작위로 10개 작품으로 한다
        # 키는 사용자 ID, 값은 추천 아이템의 ID 리스트
        pred_user2items = defaultdict(list)
        # 사용자가 이미 평가한 영화를 저장한다
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in unique_user_ids:
            user_index = user_id2index[user_id]
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = unique_movie_ids[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break
        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    RandomRecommender().run_sample()
