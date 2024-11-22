from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
import gensim
import logging
from gensim.corpora.dictionary import Dictionary

np.random.seed(0)


class LDACollaborationRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 인자 수
        factors = kwargs.get("factors", 50)
        # 에폭 수
        n_epochs = kwargs.get("n_epochs", 30)

        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
        lda_data = []
        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]
        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            lda_data.append(data["movie_id"].apply(str).tolist())

        common_dictionary = Dictionary(lda_data)
        common_corpus = [common_dictionary.doc2bow(text) for text in lda_data]

        lda_model = gensim.models.LdaModel(
            common_corpus, id2word=common_dictionary, num_topics=factors, passes=n_epochs
        )
        lda_topics = lda_model[common_corpus]

        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        pred_user2items = defaultdict(list)
        for i, (user_id, data) in enumerate(movielens_train_high_rating.groupby("user_id")):
            evaluated_movie_ids = user_evaluated_movies[user_id]

            user_topic = sorted(lda_topics[i], key=lambda x: -x[1])[0][0]
            topic_movies = lda_model.get_topic_terms(user_topic, topn=len(dataset.item_content))

            for token_id, score in topic_movies:
                movie_id = int(common_dictionary.id2token[token_id])
                if movie_id not in evaluated_movie_ids:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        # LDA에서는 평갓값 예측이 어려우므로 rmse 평가는 수행하지 않는다(편의상, 테스트 데이터를 그대로 반환한다).
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    LDACollaborationRecommender().run_sample()
