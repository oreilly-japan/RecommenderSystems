from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
import numpy as np
import gensim


np.random.seed(0)


class Item2vecRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 因子数
        factors = kwargs.get("factors", 100)
        # エポック数
        n_epochs = kwargs.get("n_epochs", 30)
        # windowサイズ
        window = kwargs.get("window", 100)
        # スキップグラム
        use_skip_gram = kwargs.get("use_skip_gram", 1)
        # 階層的ソフトマックス
        use_hierarchial_softmax = kwargs.get("use_hierarchial_softmax", 0)
        # 使用する単語の出現回数のしきい値
        min_count = kwargs.get("min_count", 5)

        item2vec_data = []
        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]
        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            # 評価された順に並び替える
            # item2vecではwindowというパラメータがあり、itemの評価された順番も重要な要素となる
            item2vec_data.append(data.sort_values("timestamp")["movie_id"].tolist())

        model = gensim.models.word2vec.Word2Vec(
            item2vec_data,
            vector_size=factors,
            window=window,
            sg=use_skip_gram,
            hs=use_hierarchial_softmax,
            epochs=n_epochs,
            min_count=min_count,
        )

        pred_user2items = dict()
        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            input_data = []
            for item_id in data.sort_values("timestamp")["movie_id"].tolist():
                if item_id in model.wv.key_to_index:
                    input_data.append(item_id)
            if len(input_data) == 0:
                # おすすめ計算できない場合は空配列
                pred_user2items[user_id] = []
                continue
            recommended_items = model.wv.most_similar(input_data, topn=10)
            pred_user2items[user_id] = [d[0] for d in recommended_items]

        # Word2vecでは評価値の予測は難しいため、rmseの評価は行わない。（便宜上、テストデータの予測値をそのまま返す）
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    Item2vecRecommender().run_sample()
