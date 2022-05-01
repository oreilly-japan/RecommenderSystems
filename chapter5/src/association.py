from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict, Counter
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

np.random.seed(0)


class AssociationRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 評価数の閾値
        min_support = kwargs.get("min_support", 0.1)
        min_threshold = kwargs.get("min_threshold", 1)

        # ユーザー×映画の行列形式に変更
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")

        # ライブラリ使用のために、4以上の評価値は1, 4未満の評価値と欠損値は0にする
        user_movie_matrix[user_movie_matrix < 4] = 0
        user_movie_matrix[user_movie_matrix.isnull()] = 0
        user_movie_matrix[user_movie_matrix >= 4] = 1

        # 支持度が高い映画
        freq_movies = apriori(user_movie_matrix, min_support=min_support, use_colnames=True)
        # アソシエーションルールの計算（リフト値の高い順に表示）
        rules = association_rules(freq_movies, metric="lift", min_threshold=min_threshold)

        # アソシエーションルールを使って、各ユーザーにまだ評価していない映画を１０本推薦する
        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        # 学習用データで評価値が4以上のものだけ取得する。
        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]

        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            # ユーザーが直近評価した５つの映画を取得
            input_data = data.sort_values("timestamp")["movie_id"].tolist()[-5:]
            # それらの映画が条件部に１本でも含まれているアソシエーションルールを抽出
            matched_flags = rules.antecedents.apply(lambda x: len(set(input_data) & x)) >= 1

            # アソシエーションルールの帰結部の映画をリストに格納し、登場頻度順に並び替え、ユーザーがまだに評価していないければ、推薦リストに追加する
            consequent_movies = []
            for i, row in rules[matched_flags].sort_values("lift", ascending=False).iterrows():
                consequent_movies.extend(row["consequents"])
            # 登場頻度をカウント
            counter = Counter(consequent_movies)
            for movie_id, movie_cnt in counter.most_common():
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                # 推薦リストが10本になったら終了する
                if len(pred_user2items[user_id]) == 10:
                    break

        # アソシエーションルールでは評価値の予測は難しいため、rmseの評価は行わない。（便宜上、テストデータの予測値をそのまま返す）
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    AssociationRecommender().run_sample()
