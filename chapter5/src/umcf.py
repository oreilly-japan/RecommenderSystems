from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np

from surprise import KNNWithMeans, Reader
from surprise import Dataset as SurpriseDataset

np.random.seed(0)


class UMCFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:

        # ピアソンの相関係数
        def peason_coefficient(u: np.ndarray, v: np.ndarray) -> float:
            u_diff = u - np.mean(u)
            v_diff = v - np.mean(v)
            numerator = np.dot(u_diff, v_diff)
            denominator = np.sqrt(sum(u_diff ** 2)) * np.sqrt(sum(v_diff ** 2))
            if denominator == 0:
                return 0.0
            return numerator / denominator

        is_naive = kwargs.get("is_naive", False)

        # 評価値をユーザー×映画の行列に変換
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")
        user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
        movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))

        # 予測対象のユーザーと映画の組
        movie_rating_predict = dataset.test.copy()
        pred_user2items = defaultdict(list)

        if is_naive:

            # 予測対象のユーザーID
            test_users = movie_rating_predict.user_id.unique()

            # 予測対象のユーザー(ユーザー1）に注目する
            for user1_id in test_users:
                similar_users = []
                similarities = []
                avgs = []

                # ユーザ−１と評価値行列中のその他のユーザー（ユーザー２）との類似度を算出する
                for user2_id in user_movie_matrix.index:
                    if user1_id == user2_id:
                        continue

                    # ユーザー１とユーザー２の評価値ベクトル
                    u_1 = user_movie_matrix.loc[user1_id, :].to_numpy()
                    u_2 = user_movie_matrix.loc[user2_id, :].to_numpy()

                    # `u_1` と `u_2` から、ともに欠損値でない要素のみ抜き出したベクトルを取得
                    common_items = ~np.isnan(u_1) & ~np.isnan(u_2)

                    # 共通して評価したアイテムがない場合はスキップ
                    if not common_items.any():
                        continue

                    u_1, u_2 = u_1[common_items], u_2[common_items]

                    # ピアソンの相関係数を使ってユーザー１とユーザー２の類似度を算出
                    rho_12 = peason_coefficient(u_1, u_2)

                    # ユーザー1との類似度が0より大きい場合、ユーザー2を類似ユーザーとみなす
                    if rho_12 > 0:
                        similar_users.append(user2_id)
                        similarities.append(rho_12)
                        avgs.append(np.mean(u_2))

                # ユーザー１の平均評価値
                avg_1 = np.mean(user_movie_matrix.loc[user1_id, :].dropna().to_numpy())

                # 予測対象の映画のID
                test_movies = movie_rating_predict[movie_rating_predict["user_id"] == user1_id].movie_id.values
                # 予測できない映画への評価値はユーザー１の平均評価値とする
                movie_rating_predict.loc[(movie_rating_predict["user_id"] == user1_id), "rating_pred"] = avg_1

                if similar_users:
                    for movie_id in test_movies:
                        if movie_id in movie_id2index:
                            r_xy = user_movie_matrix.loc[similar_users, movie_id].to_numpy()
                            rating_exists = ~np.isnan(r_xy)

                            # 類似ユーザーが対象となる映画への評価値を持っていない場合はスキップ
                            if not rating_exists.any():
                                continue

                            r_xy = r_xy[rating_exists]
                            rho_1x = np.array(similarities)[rating_exists]
                            avg_x = np.array(avgs)[rating_exists]
                            r_hat_1y = avg_1 + np.dot(rho_1x, (r_xy - avg_x)) / rho_1x.sum()

                            # 予測評価値を格納
                            movie_rating_predict.loc[
                                (movie_rating_predict["user_id"] == user1_id)
                                & (movie_rating_predict["movie_id"] == movie_id),
                                "rating_pred",
                            ] = r_hat_1y

        else:
            # Surprise用にデータを加工
            reader = Reader(rating_scale=(0.5, 5))
            data_train = SurpriseDataset.load_from_df(
                dataset.train[["user_id", "movie_id", "rating"]], reader
            ).build_full_trainset()

            sim_options = {"name": "pearson", "user_based": True}  # 類似度を計算する方法を指定する  # False にするとアイテムベースとなる
            knn = KNNWithMeans(k=30, min_k=1, sim_options=sim_options)
            knn.fit(data_train)

            # 学習データセットで評価値のないユーザーとアイテムの組み合わせを準備
            data_test = data_train.build_anti_testset(None)
            predictions = knn.test(data_test)

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

            pred_user2items = get_top_n(predictions, n=10)

            average_score = dataset.train.rating.mean()
            pred_results = []
            for _, row in dataset.test.iterrows():
                user_id = row["user_id"]
                movie_id = row["movie_id"]
                # 学習データに存在せずテストデータにしか存在しないユーザーや映画についての予測評価値は、全体の平均評価値とする
                if user_id not in user_id2index or movie_id not in movie_id2index:
                    pred_results.append(average_score)
                    continue
                pred_score = knn.predict(uid=user_id, iid=movie_id).est
                pred_results.append(pred_score)
            movie_rating_predict["rating_pred"] = pred_results

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    UMCFRecommender().run_sample()
