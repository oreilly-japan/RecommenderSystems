from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import itertools
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR

np.random.seed(0)


class RFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 評価値をユーザー×映画の行列に変換。欠損値は、平均値または０で穴埋めする
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")
        user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
        movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))

        # 学習に用いる学習用データ中のユーザーと映画の組を取得する
        train_keys = dataset.train[["user_id", "movie_id"]]
        # 学習用データ中の評価値を学習の正解データとして取得する
        train_y = dataset.train.rating.values

        # 評価値を予測したいテスト用データ中のユーザーと映画の組を取得する
        test_keys = dataset.test[["user_id", "movie_id"]]
        # ランキング形式の推薦リスト作成のために学習用データに存在するすべてのユーザーとすべての映画の組み合わせを取得する
        train_all_keys = user_movie_matrix.stack(dropna=False).reset_index()[["user_id", "movie_id"]]

        # 特徴量を作成する
        train_x = train_keys.copy()
        test_x = test_keys.copy()
        train_all_x = train_all_keys.copy()

        # 学習用データに存在するユーザーごとの評価値の最小値、最大値、平均値
        # 及び、映画ごとの評価値の最小値、最大値、平均値を特徴量として追加
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
        # テスト用データにしか存在しないユーザーや映画の特徴量を、学習用データ全体の平均評価値で埋める
        average_rating = train_y.mean()
        test_x.fillna(average_rating, inplace=True)

        # 映画が特定の genre であるかどうかを表す特徴量を追加
        movie_genres = dataset.item_content[["movie_id", "genre"]]
        genres = set(list(itertools.chain(*movie_genres.genre)))
        for genre in genres:
            movie_genres[f"is_{genre}"] = movie_genres.genre.apply(lambda x: genre in x)
        movie_genres.drop("genre", axis=1, inplace=True)
        train_x = train_x.merge(movie_genres, on="movie_id")
        test_x = test_x.merge(movie_genres, on="movie_id")
        train_all_x = train_all_x.merge(movie_genres, on="movie_id")

        # 特徴量としては使わない情報を削除
        train_x = train_x.drop(columns=["user_id", "movie_id"])
        test_x = test_x.drop(columns=["user_id", "movie_id"])
        train_all_x = train_all_x.drop(columns=["user_id", "movie_id"])

        # Random Forest を用いた学習
        reg = RFR(n_jobs=-1, random_state=0)
        reg.fit(train_x.values, train_y)

        # テスト用データ内のユーザーと映画の組に対して評価値を予測する
        test_pred = reg.predict(test_x.values)

        movie_rating_predict = test_keys.copy()
        movie_rating_predict["rating_pred"] = test_pred

        # 学習用データに存在するすべてのユーザーとすべての映画の組み合わせに対して評価値を予測する
        train_all_pred = reg.predict(train_all_x.values)

        pred_train_all = train_all_keys.copy()
        pred_train_all["rating_pred"] = train_all_pred
        pred_matrix = pred_train_all.pivot(index="user_id", columns="movie_id", values="rating_pred")

        # ユーザーが学習用データ内で評価していない映画の中から
        # 予測評価値が高い順に10件の映画をランキング形式の推薦リストとする
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
