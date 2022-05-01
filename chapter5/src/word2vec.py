from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
import numpy as np
import gensim


np.random.seed(0)


class Word2vecRecommender(BaseRecommender):
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

        movie_content = dataset.item_content.copy()
        # tagが付与されていない映画もあるが、genreはすべての映画に付与されている
        # tagとgenreを結合したものを映画のコンテンツ情報として似ている映画を探して推薦していく
        # tagがない映画に関しては、NaNになっているので、空のリストに変換してから処理をする
        movie_content["tag_genre"] = movie_content["tag"].fillna("").apply(list) + movie_content["genre"].apply(list)
        movie_content["tag_genre"] = movie_content["tag_genre"].apply(lambda x: set(map(str, x)))

        # タグとジャンルデータを使って、word2vecを学習する
        tag_genre_data = movie_content.tag_genre.tolist()
        model = gensim.models.word2vec.Word2Vec(
            tag_genre_data,
            vector_size=factors,
            window=window,
            sg=use_skip_gram,
            hs=use_hierarchial_softmax,
            epochs=n_epochs,
            min_count=min_count,
        )

        # 各映画のベクトルを計算する
        # 各映画に付与されているタグ・ジャンルのベクトルの平均を映画のベクトルとする
        movie_vectors = []
        tag_genre_in_model = set(model.wv.key_to_index.keys())

        titles = []
        ids = []

        for i, tag_genre in enumerate(tag_genre_data):
            # word2vecのモデルで使用可能なタグ・ジャンルに絞る
            input_tag_genre = set(tag_genre) & tag_genre_in_model
            if len(input_tag_genre) == 0:
                # word2vecに基づいてベクトル計算できない映画にはランダムのベクトルを付与
                vector = np.random.randn(model.vector_size)
            else:
                vector = model.wv[input_tag_genre].mean(axis=0)
            titles.append(movie_content.iloc[i]["title"])
            ids.append(movie_content.iloc[i]["movie_id"])
            movie_vectors.append(vector)

        # 後続の類似度計算がしやすいように、numpyの配列で保持しておく
        movie_vectors = np.array(movie_vectors)

        # 正規化したベクトル
        sum_vec = np.sqrt(np.sum(movie_vectors ** 2, axis=1))
        movie_norm_vectors = movie_vectors / sum_vec.reshape((-1, 1))

        def find_similar_items(vec, evaluated_movie_ids, topn=10):
            score_vec = np.dot(movie_norm_vectors, vec)
            similar_indexes = np.argsort(-score_vec)
            similar_items = []
            for similar_index in similar_indexes:
                if ids[similar_index] not in evaluated_movie_ids:
                    similar_items.append(ids[similar_index])
                if len(similar_items) == topn:
                    break
            return similar_items

        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        id2index = dict(zip(ids, range(len(ids))))
        pred_user2items = dict()
        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            evaluated_movie_ids = user_evaluated_movies[user_id]
            movie_ids = data.sort_values("timestamp")["movie_id"].tolist()[-5:]

            movie_indexes = [id2index[id] for id in movie_ids]
            user_vector = movie_norm_vectors[movie_indexes].mean(axis=0)
            recommended_items = find_similar_items(user_vector, evaluated_movie_ids, topn=10)
            pred_user2items[user_id] = recommended_items

        # Word2vecでは評価値の予測は難しいため、rmseの評価は行わない。（便宜上、テストデータの予測値をそのまま返す）
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    Word2vecRecommender().run_sample()
