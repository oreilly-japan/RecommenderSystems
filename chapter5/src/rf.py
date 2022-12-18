from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import itertools
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR

np.random.seed(0)


class RFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 평갓값을 사용자 x 영화의 행렬로 변환한다. 결손값은 평균값 또는 0으로 채운다
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")
        user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
        movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))

        # 학습에 사용하는 학습용 데이터 중 사용자와 영화의 조합을 얻는다
        train_keys = dataset.train[["user_id", "movie_id"]]
        # 학습용 데이터 중의 평갓값을 학습의 정답 데이터로서 얻는다
        train_y = dataset.train.rating.values

        # 평갓값을 예측하고자 하는 테스트용 데이터 안의 사용자와 영화의 조합을 얻는다
        test_keys = dataset.test[["user_id", "movie_id"]]
        # 순위 형식의 추천 리스르 작성을 윟 학습용 데이터에 존재하는 모든 사용자와 모든 영화의 조합을 저장한다
        train_all_keys = user_movie_matrix.stack(dropna=False).reset_index()[["user_id", "movie_id"]]

        # 특징량을 작성한다
        train_x = train_keys.copy()
        test_x = test_keys.copy()
        train_all_x = train_all_keys.copy()

        # 학습용 데이터에 존재하는 사용자별 평갓값의 최솟값, 최댓값, 평균값
        # 及び、映画ごとの評価値の最小値、最大値、平均値を特徴量として追加
        # 및, 영화별 평갓값의 최솟값, 최댓값, 평균값을 특징량ㅇ로 추가한다
        aggregators = ["min", "max", "mean"]
        user_features = dataset.train.groupby("user_id").rating.agg(aggregators).to_dict()
        movie_features = dataset.train.groupby("movie_id").rating.agg(aggregators).to_dict()
        for agg in aggregators:
            train_x[f"u_{agg}"] = train_x["user_id"].map(user_features[agg])
            test_x[f"u_{agg}"] = test_x["user_id"].map(user_features[agg])
            train_all_x[f"u_{agg}"] = train_all_x["user_id"].map(user_features[agg])
            train_x[f"m_{agg}"] = train_x["movie_id"].map(movie_features[agg])
            test_x[f"m_{agg}"] = test_x["movie_id"].map(movie_features[agg])
            train_all_x[f"m_{agg}"] = train_all_x["movie_id"].map(movie_features[agg])
        # 테스트용 데이터에만 존재하는 사용자나 영화의 특징량을, 학습용 데이터 전체의 평균 평갓값으로 채운다
        average_rating = train_y.mean()
        test_x.fillna(average_rating, inplace=True)

        # 영화가 특정한 genre에 있는지를 나타태는 특징량을 추가
        movie_genres = dataset.item_content[["movie_id", "genre"]]
        genres = set(list(itertools.chain(*movie_genres.genre)))
        for genre in genres:
            movie_genres[f"is_{genre}"] = movie_genres.genre.apply(lambda x: genre in x)
        movie_genres.drop("genre", axis=1, inplace=True)
        train_x = train_x.merge(movie_genres, on="movie_id")
        test_x = test_x.merge(movie_genres, on="movie_id")
        train_all_x = train_all_x.merge(movie_genres, on="movie_id")

        # 특징량으로서는 사용하지 않늦 정보를 삭제
        train_x = train_x.drop(columns=["user_id", "movie_id"])
        test_x = test_x.drop(columns=["user_id", "movie_id"])
        train_all_x = train_all_x.drop(columns=["user_id", "movie_id"])

        # Random Forest를 사용한 학습
        reg = RFR(n_jobs=-1, random_state=0)
        reg.fit(train_x.values, train_y)

        # 테스트용 데이터 안의 사용자와 영화의 조합에 대한 평갓값을 예측한다
        test_pred = reg.predict(test_x.values)

        movie_rating_predict = test_keys.copy()
        movie_rating_predict["rating_pred"] = test_pred

        # 학습용 데이터에 존재하는 모든 사용자와 모든 영화의 조합애 대해 평갓값을 예측한다
        train_all_pred = reg.predict(train_all_x.values)

        pred_train_all = train_all_keys.copy()
        pred_train_all["rating_pred"] = train_all_pred
        pred_matrix = pred_train_all.pivot(index="user_id", columns="movie_id", values="rating_pred")

        # 사용자가 학습용 데이터 안에서 평가하지 않은 영화 중에서
        # 예측 평갓값이 높은 순으로 10건의 영화를 순위 형식의 추천 리스트로 한다
        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in dataset.train.user_id.unique():
            movie_indexes = np.argsort(-pred_matrix.loc[user_id, :]).values
            for movie_index in movie_indexes:
                movie_id = user_movie_matrix.columns[movie_index]
                if movie_id not in (user_evaluated_movies[user_id]):
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    RFRecommender().run_sample()
