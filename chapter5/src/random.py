from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np

np.random.seed(0)


class RandomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # ユーザーIDとアイテムIDに対して、０始まりのインデックスを割り振る
        unique_user_ids = sorted(dataset.train.user_id.unique())
        unique_movie_ids = sorted(dataset.train.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        # ユーザー×アイテムの行列で、各セルの予測評価値は0.5〜5.0の一様乱数とする
        pred_matrix = np.random.uniform(0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids)))

        # rmse評価用に、テストデータに出てくるユーザーとアイテムの予測評価値を格納する
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            # テストデータのアイテムIDが学習用に登場していない場合も乱数を格納する
            if row["movie_id"] not in movie_id2index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue
            # テストデータに現れるユーザーIDとアイテムIDのインデックスを取得し、評価値行列の値を取得する
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        # ランキング評価用のデータ作成
        # 各ユーザに対するおすすめ映画は、そのユーザがまだ評価していない映画の中からランダムに10作品とする
        # キーはユーザーIDで、バリューはおすすめのアイテムIDのリスト
        pred_user2items = defaultdict(list)
        # ユーザーがすでに評価した映画を取得する
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
