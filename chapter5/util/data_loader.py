import pandas as pd
import os
from util.models import Dataset


class DataLoader:
    def __init__(self, num_users: int = 1000, num_test_items: int = 5, data_path: str = "../data/ml-10M100K/"):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path

    def load(self) -> Dataset:
        ratings, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(ratings)
        # ranking 용 평가 데이터는 각 사용자의 평갓값이 4 이상인 영화만을 정답으로 한다
        # 키는 사용자 ID, 값은 사용자가 높이 평가한 아이템의 ID 리스트
        movielens_test_user2items = (
            movielens_test[movielens_test.rating >= 4].groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        )
        return Dataset(movielens_train, movielens_test, movielens_test_user2items, movie_content)

    def _split_data(self, movielens: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        # 학습용과 테스트용으로 데이터를 분할한다
        # 각 사용자의 직전 5개 영화를 평가용으로 사용하고, 그 이외는 학습용으로 한다
        # 먼저, 각 사용자가 평가한 영화의 순서를 계산한다
        # 최근 부여한 영화부터 순서를 부여한다(0부터 시작)
        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(ascending=False, method="first")
        movielens_train = movielens[movielens["rating_order"] > self.num_test_items]
        movielens_test = movielens[movielens["rating_order"] <= self.num_test_items]
        return movielens_train, movielens_test

    def _load(self) -> (pd.DataFrame, pd.DataFrame):
        # 영화 정보 로딩(10197 작품)
        # movie_id와 제목만 사용
        m_cols = ["movie_id", "title", "genre"]
        movies = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"), names=m_cols, sep="::", encoding="latin-1", engine="python"
        )
        # genre를 list 형식으로 저장한다
        movies["genre"] = movies.genre.apply(lambda x: x.split("|"))

        # 사용자가 부여한 영화의 태그 정보를 로딩한다
        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        user_tagged_movies = pd.read_csv(
            os.path.join(self.data_path, "tags.dat"), names=t_cols, sep="::", engine="python"
        )
        # tag를 소문자로 한다
        user_tagged_movies["tag"] = user_tagged_movies["tag"].str.lower()
        movie_tags = user_tagged_movies.groupby("movie_id").agg({"tag": list})

        # 태그 정보를 결합한다
        movies = movies.merge(movie_tags, on="movie_id", how="left")

        # 평가 데이터를 로딩한다
        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(os.path.join(self.data_path, "ratings.dat"), names=r_cols, sep="::", engine="python")

        # user 수를 num_users로 줄인다
        valid_user_ids = sorted(ratings.user_id.unique())[: self.num_users]
        ratings = ratings[ratings.user_id <= max(valid_user_ids)]

        # 위 데이터를 결합한다
        movielens_ratings = ratings.merge(movies, on="movie_id")

        return movielens_ratings, movies
