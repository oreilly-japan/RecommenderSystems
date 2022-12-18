from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
import numpy as np
import gensim


np.random.seed(0)


class Word2vecRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 인자 수
        factors = kwargs.get("factors", 100)
        # 에폭 수
        n_epochs = kwargs.get("n_epochs", 30)
        # window 크기
        window = kwargs.get("window", 100)
        # 스킵 그램
        use_skip_gram = kwargs.get("use_skip_gram", 1)
        # 계층적 소프트맥스
        use_hierarchial_softmax = kwargs.get("use_hierarchial_softmax", 0)
        # 사용한 단어의 출현 횟수의 임곗값
        min_count = kwargs.get("min_count", 5)

        movie_content = dataset.item_content.copy()
        # tag가 부여되지 않은 영화는 있지만, genre는 모든 영화에 부여되어 있다
        # tag와 genre를 결합한 것을 영화 콘텐츠 정보로해서 비슷한 영화를 찾아 추천한다
        # tag가 없는 영화의 경우에는 NaN으로 되어 있으므로, 빈 리스트로 변환한 뒤 처리한다
        movie_content["tag_genre"] = movie_content["tag"].fillna("").apply(list) + movie_content["genre"].apply(list)
        movie_content["tag_genre"] = movie_content["tag_genre"].apply(lambda x: set(map(str, x)))

        # 태그와 장르 데이터를 사용해 word2vec을 학습한다
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

        # 각 영화의 벡터를 계산한다
        # 각 영화에 부여되어 있는 태그/장르의 벡트 평균을 영화 벡터로 한다
        movie_vectors = []
        tag_genre_in_model = set(model.wv.key_to_index.keys())

        titles = []
        ids = []

        for i, tag_genre in enumerate(tag_genre_data):
            # word2vec 모델에서 사용할 수 있는 태그/장르로 한정한다
            input_tag_genre = set(tag_genre) & tag_genre_in_model
            if len(input_tag_genre) == 0:
                # word2vec에 기반해 벡터 계산할 수 없는 형화에는 무작위 벡터를 부여한다
                vector = np.random.randn(model.vector_size)
            else:
                vector = model.wv[input_tag_genre].mean(axis=0)
            titles.append(movie_content.iloc[i]["title"])
            ids.append(movie_content.iloc[i]["movie_id"])
            movie_vectors.append(vector)

        # 후속 유사도 계산을 쉽게할 수 있도록 numpy 배열로 저장해 둔다
        movie_vectors = np.array(movie_vectors)

        # 정규화 벡터
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

        # Word2vec에서는 평갓값 예측이 어려우므로, rmse의 평가는 수행하지 않는다(편의상, 테스트 데이터의 예측값을 그대로 반환한다)
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    Word2vecRecommender().run_sample()
