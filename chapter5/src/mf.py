from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
from surprise import SVD, Reader
import pandas as pd
from surprise import Dataset as SurpriseDataset

np.random.seed(0)


class MFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 인자 수
        factors = kwargs.get("factors", 5)
        # 평갓값의 임곗값
        minimum_num_rating = kwargs.get("minimum_num_rating", 100)
        # 바이어스 항 사용
        use_biase = kwargs.get("use_biase", False)
        # 학습률
        lr_all = kwargs.get("lr_all", 0.005)
        # 에폭 수
        n_epochs = kwargs.get("n_epochs", 50)

        # 평갓값이 minimum_num_rating건 이상 있는 연화로 필터링한다
        filtered_movielens_train = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= minimum_num_rating
        )

        # Surprise용으로 데이터를 가공
        reader = Reader(rating_scale=(0.5, 5))
        data_train = SurpriseDataset.load_from_df(
            filtered_movielens_train[["user_id", "movie_id", "rating"]], reader
        ).build_full_trainset()

        # Surprise로 행렬 분새를 학습
        # SVD라는 이름을 사용하지만, 특이점 분해가 아니라 Matrix Factorization이 실행된다
        matrix_factorization = SVD(n_factors=factors, n_epochs=n_epochs, lr_all=lr_all, biased=use_biase)
        matrix_factorization.fit(data_train)

        def get_top_n(predictions, n=10):
            # 각 사용자별로 예측된 아이템을 저장한다
            top_n = defaultdict(list)
            for uid, iid, true_r, est, _ in predictions:
                top_n[uid].append((iid, est))

            # 사용자별로 아이템을 예측 평값값순으로 나열하고 상위 n개를 저장한다
            for uid, user_ratings in top_n.items():
                user_ratings.sort(key=lambda x: x[1], reverse=True)
                top_n[uid] = [d[0] for d in user_ratings[:n]]

            return top_n

        # 학습 데이터에 나오지 않은 사용자와 아이템의 조합을 준비한다
        data_test = data_train.build_anti_testset(None)
        predictions = matrix_factorization.test(data_test)
        pred_user2items = get_top_n(predictions, n=10)

        test_data = pd.DataFrame.from_dict(
            [{"user_id": p.uid, "movie_id": p.iid, "rating_pred": p.est} for p in predictions]
        )
        movie_rating_predict = dataset.test.merge(test_data, on=["user_id", "movie_id"], how="left")

        # 예측할 수 없는 위치에는 평균값을 저장한다
        movie_rating_predict.rating_pred.fillna(filtered_movielens_train.rating.mean(), inplace=True)

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    MFRecommender().run_sample()
