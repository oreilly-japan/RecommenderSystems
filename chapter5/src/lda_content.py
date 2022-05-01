from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
import gensim
import logging
from gensim.corpora.dictionary import Dictionary
from collections import Counter

np.random.seed(0)


class LDAContentRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 因子数
        factors = kwargs.get("factors", 50)
        # エポック数
        n_epochs = kwargs.get("n_epochs", 30)

        movie_content = dataset.item_content.copy()
        # tagが付与されていない映画もあるが、genreはすべての映画に付与されている
        # tagとgenreを結合したものを映画のコンテンツ情報として似ている映画を探して推薦していく
        # tagがない映画に関しては、NaNになっているので、空のリストに変換してから処理をする
        movie_content["tag_genre"] = movie_content["tag"].fillna("").apply(list) + movie_content["genre"].apply(list)
        movie_content["tag_genre"] = movie_content["tag_genre"].apply(lambda x: list(map(str, x)))

        # タグとジャンルデータを使って、word2vecを学習する
        tag_genre_data = movie_content.tag_genre.tolist()

        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
        common_dictionary = Dictionary(tag_genre_data)
        common_corpus = [common_dictionary.doc2bow(text) for text in tag_genre_data]

        lda_model = gensim.models.LdaModel(
            common_corpus, id2word=common_dictionary, num_topics=factors, passes=n_epochs
        )
        lda_topics = lda_model[common_corpus]
        movie_topics = []
        movie_topic_scores = []
        for movie_index, lda_topic in enumerate(lda_topics):
            sorted_topic = sorted(lda_topics[movie_index], key=lambda x: -x[1])
            movie_topic, topic_score = sorted_topic[0]
            movie_topics.append(movie_topic)
            movie_topic_scores.append(topic_score)
        movie_content["topic"] = movie_topics
        movie_content["topic_score"] = movie_topic_scores

        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        movie_id2index = dict(zip(movie_content.movie_id.tolist(), range(len(movie_content))))
        pred_user2items = defaultdict(list)
        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            evaluated_movie_ids = user_evaluated_movies[user_id]
            movie_ids = data.sort_values("timestamp")["movie_id"].tolist()[-10:]

            movie_indexes = [movie_id2index[id] for id in movie_ids]

            topic_counter = Counter([movie_topics[i] for i in movie_indexes])
            frequent_topic = topic_counter.most_common(1)[0][0]
            topic_movies = (
                movie_content[movie_content.topic == frequent_topic]
                .sort_values("topic_score", ascending=False)
                .movie_id.tolist()
            )

            for movie_id in topic_movies:
                if movie_id not in evaluated_movie_ids:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        # LDAでは評価値の予測は難しいため、rmseの評価は行わない。（便宜上、テストデータの予測値をそのまま返す）
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    LDAContentRecommender().run_sample()
