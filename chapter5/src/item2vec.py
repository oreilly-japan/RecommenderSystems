from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
import numpy as np
import gensim


np.random.seed(0)


class Item2vecRecommender(BaseRecommender):
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
        # 사용할 단어의 출현 횟수의 임곗값
        min_count = kwargs.get("min_count", 5)

        item2vec_data = []
        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]
        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            # 평가된 순으로 나열한다
            # item2vec에서는 window라는 파라미터가 있으며, item의 평가된 순서도 중요한 요소가 된다
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
                # 추천 계싼할 수 없는 경우에는 빈 배열
                pred_user2items[user_id] = []
                continue
            recommended_items = model.wv.most_similar(input_data, topn=10)
            pred_user2items[user_id] = [d[0] for d in recommended_items]

        # Word2vec에서는 평갓값 예측이 어려우므로, rmse는 평가하지 않는다(편의상, 테스트 데이터의 예측값을 그대로 반환한다).
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    Item2vecRecommender().run_sample()
