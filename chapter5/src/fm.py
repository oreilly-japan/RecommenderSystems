from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
import xlearn as xl
from sklearn.feature_extraction import DictVectorizer

np.random.seed(0)


class FMRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 인자 수
        factors = kwargs.get("factors", 10)
        # 평갓값의 임곗값
        minimum_num_rating = kwargs.get("minimum_num_rating", 200)
        # 에폭 수
        n_epochs = kwargs.get("n_epochs", 50)
        # 학습률
        lr = kwargs.get("lr", 0.01)
        # 보충 정보 사용
        use_side_information = kwargs.get("use_side_information", False)

        # 평갓값이 minimum_num_rating건 이상인 영화로 필터링한다
        filtered_movielens_train = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= minimum_num_rating
        )

        # 사용자가 평가한 영화
        user_evaluated_movies = (
            filtered_movielens_train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        )

        train_data_for_fm = []
        y = []
        for i, row in filtered_movielens_train.iterrows():
            x = {"user_id": str(row["user_id"]), "movie_id": str(row["movie_id"])}
            if use_side_information:
                x["tag"] = row["tag"]
                x["user_rating_avg"] = np.mean(user_evaluated_movies[row["user_id"]])
            train_data_for_fm.append(x)
            y.append(row["rating"])

        y = np.array(y)

        vectorizer = DictVectorizer()
        X = vectorizer.fit_transform(train_data_for_fm).toarray()

        fm_model = xl.FMModel(task="reg", metric="rmse", lr=lr, opt="sgd", k=factors, epoch=n_epochs)

        # Start to train
        fm_model.fit(X, y, is_lock_free=False)

        unique_user_ids = sorted(filtered_movielens_train.user_id.unique())
        unique_movie_ids = sorted(filtered_movielens_train.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        test_data_for_fm = []
        for user_id in unique_user_ids:
            for movie_id in unique_movie_ids:
                x = {"user_id": str(user_id), "movie_id": str(movie_id)}
                if use_side_information:
                    tag = dataset.item_content[dataset.item_content.movie_id == movie_id].tag.tolist()[0]
                    x["tag"] = tag
                    x["user_rating_avg"] = np.mean(user_evaluated_movies[row["user_id"]])
                test_data_for_fm.append(x)

        X_test = vectorizer.transform(test_data_for_fm).toarray()
        y_pred = fm_model.predict(X_test)
        pred_matrix = y_pred.reshape(len(unique_user_ids), len(unique_movie_ids))

        # 학습용에 나오지 않은 사용자나 영화의 예측 평갓값은 평균 평간값으로 한다
        average_score = dataset.train.rating.mean()
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            if user_id not in user_id2index or row["movie_id"] not in movie_id2index:
                pred_results.append(average_score)
                continue
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        pred_user2items = defaultdict(list)

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
    FMRecommender().run_sample()
