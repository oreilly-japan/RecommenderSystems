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
        # 因子数
        factors = kwargs.get("factors", 5)
        # 評価数の閾値
        minimum_num_rating = kwargs.get("minimum_num_rating", 100)
        # バイアス項の使用
        use_biase = kwargs.get("use_biase", False)
        # 学習率
        lr_all = kwargs.get("lr_all", 0.005)
        # エポック数
        n_epochs = kwargs.get("n_epochs", 50)

        # 評価数がminimum_num_rating件以上ある映画に絞る
        filtered_movielens_train = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= minimum_num_rating
        )

        # Surprise用にデータを加工
        reader = Reader(rating_scale=(0.5, 5))
        data_train = SurpriseDataset.load_from_df(
            filtered_movielens_train[["user_id", "movie_id", "rating"]], reader
        ).build_full_trainset()

        # Surpriseで行列分解を学習
        # SVDという名前だが、特異値分解ではなく、Matrix Factorizationが実行される
        matrix_factorization = SVD(n_factors=factors, n_epochs=n_epochs, lr_all=lr_all, biased=use_biase)
        matrix_factorization.fit(data_train)

        def get_top_n(predictions, n=10):
            # 各ユーザーごとに、予測されたアイテムを格納する
            top_n = defaultdict(list)
            for uid, iid, true_r, est, _ in predictions:
                top_n[uid].append((iid, est))

            # ユーザーごとに、アイテムを予測評価値順に並べ上位n個を格納する
            for uid, user_ratings in top_n.items():
                user_ratings.sort(key=lambda x: x[1], reverse=True)
                top_n[uid] = [d[0] for d in user_ratings[:n]]

            return top_n

        # 学習データに出てこないユーザーとアイテムの組み合わせを準備
        data_test = data_train.build_anti_testset(None)
        predictions = matrix_factorization.test(data_test)
        pred_user2items = get_top_n(predictions, n=10)

        test_data = pd.DataFrame.from_dict(
            [{"user_id": p.uid, "movie_id": p.iid, "rating_pred": p.est} for p in predictions]
        )
        movie_rating_predict = dataset.test.merge(test_data, on=["user_id", "movie_id"], how="left")

        # 予測ができない箇所には、平均値を格納する
        movie_rating_predict.rating_pred.fillna(filtered_movielens_train.rating.mean(), inplace=True)

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    MFRecommender().run_sample()
