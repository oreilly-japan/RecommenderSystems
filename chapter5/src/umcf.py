from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np

from surprise import KNNWithMeans, Reader
from surprise import Dataset as SurpriseDataset

np.random.seed(0)


class UMCFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:

        # 피어슨 상관 계수
        def peason_coefficient(u: np.ndarray, v: np.ndarray) -> float:
            u_diff = u - np.mean(u)
            v_diff = v - np.mean(v)
            numerator = np.dot(u_diff, v_diff)
            denominator = np.sqrt(sum(u_diff ** 2)) * np.sqrt(sum(v_diff ** 2))
            if denominator == 0:
                return 0.0
            return numerator / denominator

        is_naive = kwargs.get("is_naive", False)

        # 평갓값을 사용자 x 영화 행렬로 변환한다
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")
        user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
        movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))

        # 예측 대상 사용자와 영화 그룹
        movie_rating_predict = dataset.test.copy()
        pred_user2items = defaultdict(list)

        if is_naive:

            # 예측 대상 사용자 ID
            test_users = movie_rating_predict.user_id.unique()

            # 예측 대상 사요자(사용자 1)에 주목한다
            for user1_id in test_users:
                similar_users = []
                similarities = []
                avgs = []

                # 사용자 1과 평갓값 행렬 안의 다른 사용자(사용자 2)와의 유사도를 산출한다
                for user2_id in user_movie_matrix.index:
                    if user1_id == user2_id:
                        continue

                    # 사용자 1과 사용자 2의 평갓값 벡터
                    u_1 = user_movie_matrix.loc[user1_id, :].to_numpy()
                    u_2 = user_movie_matrix.loc[user2_id, :].to_numpy()

                    # `u_1`과 `u_2` 모두에서 결손값이 없는 요소만 추출한 벡터를 얻는다
                    common_items = ~np.isnan(u_1) & ~np.isnan(u_2)

                    # 공통으로 평가한 아이템이 없는 경우는 스킵한다
                    if not common_items.any():
                        continue

                    u_1, u_2 = u_1[common_items], u_2[common_items]

                    # 피어슨 상관 계수를 사용해 사용자 1과 사용자 2의 유사도를 산출한다
                    rho_12 = peason_coefficient(u_1, u_2)

                    # 사용자 1과의 유사도가 0보다 큰 경우, 사용자 2를 유사 사용자로 간주한다
                    if rho_12 > 0:
                        similar_users.append(user2_id)
                        similarities.append(rho_12)
                        avgs.append(np.mean(u_2))

                # 사용자 1의 평균 평갓값
                avg_1 = np.mean(user_movie_matrix.loc[user1_id, :].dropna().to_numpy())

                # 예측 대상의 영화 ID
                test_movies = movie_rating_predict[movie_rating_predict["user_id"] == user1_id].movie_id.values
                # 예측할 수 없는 영화에 대한 평갓값은 사용자 1의 평균 평갓값으로 한다
                movie_rating_predict.loc[(movie_rating_predict["user_id"] == user1_id), "rating_pred"] = avg_1

                if similar_users:
                    for movie_id in test_movies:
                        if movie_id in movie_id2index:
                            r_xy = user_movie_matrix.loc[similar_users, movie_id].to_numpy()
                            rating_exists = ~np.isnan(r_xy)

                            # 유사 사용자가 대상이 되는 영화에 대한 평갓값을 갖지 않는 경우는 스킵한다
                            if not rating_exists.any():
                                continue

                            r_xy = r_xy[rating_exists]
                            rho_1x = np.array(similarities)[rating_exists]
                            avg_x = np.array(avgs)[rating_exists]
                            r_hat_1y = avg_1 + np.dot(rho_1x, (r_xy - avg_x)) / rho_1x.sum()

                            # 예측 평갓값을 저장한다
                            movie_rating_predict.loc[
                                (movie_rating_predict["user_id"] == user1_id)
                                & (movie_rating_predict["movie_id"] == movie_id),
                                "rating_pred",
                            ] = r_hat_1y

        else:
            # Surprise용으로 데이터를 가공한다
            reader = Reader(rating_scale=(0.5, 5))
            data_train = SurpriseDataset.load_from_df(
                dataset.train[["user_id", "movie_id", "rating"]], reader
            ).build_full_trainset()

            sim_options = {"name": "pearson", "user_based": True}  # 유사도를 계산하는 방법을 지정한다  # False로 하면 아이템 기반이 된다
            knn = KNNWithMeans(k=30, min_k=1, sim_options=sim_options)
            knn.fit(data_train)

            # 학습 데이터셋에서 평갓값이 없는 사용자와 아이템의 조합을 준비
            data_test = data_train.build_anti_testset(None)
            predictions = knn.test(data_test)

            def get_top_n(predictions, n=10):
                # 각 사용자별로 예측된 아이템을 저장한다
                top_n = defaultdict(list)
                for uid, iid, true_r, est, _ in predictions:
                    top_n[uid].append((iid, est))

                # 상요자별로 아이템을 예측 평갓값순으로 나열하고 상위 n개를 저장한다
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
                # 학습 데이터에 존재하지 않고 테스트 데이터에만 존재하는 사용자나 영화에 관한 예측 평갓값는 전체 평균 평갓값으로 한다
                if user_id not in user_id2index or movie_id not in movie_id2index:
                    pred_results.append(average_score)
                    continue
                pred_score = knn.predict(uid=user_id, iid=movie_id).est
                pred_results.append(pred_score)
            movie_rating_predict["rating_pred"] = pred_results

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    UMCFRecommender().run_sample()
