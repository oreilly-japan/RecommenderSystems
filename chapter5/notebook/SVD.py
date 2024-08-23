# 特異値分解(SVD)
# 親のフォルダのパスを追加
import sys; sys.path.insert(0, '..')

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
# Movielensのデータの読み込み
data_loader = DataLoader(num_users=1000, num_test_items=5, data_path='../data/ml-10M100K/')
movielens = data_loader.load()
user_movie_matrix = movielens.train.pivot(index='user_id', columns='movie_id', values='rating')
user_movie_matrix
# スパース情報
user_num = len(user_movie_matrix.index)
item_num = len(user_movie_matrix.columns)
non_null_num = user_num*item_num - user_movie_matrix.isnull().sum().sum()
non_null_ratio = non_null_num / (user_num*item_num)

print(f'ユーザー数={user_num}, アイテム数={item_num}, 密度={non_null_ratio:.2f}')
user_movie_matrix.fillna(0)
import scipy
import numpy as np

# 評価値をユーザー×映画の行列に変換。欠損値は、平均値で穴埋めする
user_movie_matrix = movielens.train.pivot(index='user_id', columns='movie_id', values='rating')
user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))
matrix = user_movie_matrix.fillna(movielens.train.rating.mean()).to_numpy()


# 因子数kで特異値分解を行う
P, S, Qt = scipy.sparse.linalg.svds(matrix, k=5)

# 予測評価値行列
pred_matrix = np.dot(np.dot(P, np.diag(S)), Qt)

print(f"P: {P.shape}, S: {S.shape}, Qt: {Qt.shape}, pred_matrix: {pred_matrix.shape}")
# SVDレコメンド
from src.svd import SVDRecommender
recommender = SVDRecommender()
recommend_result = recommender.recommend(movielens)
#  評価
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
print(metrics)
# 欠損値のを平均値で穴埋め
recommend_result = recommender.recommend(movielens, fillna_with_zero=False)
metrics = metric_calculator.calc(
movielens.test.rating.tolist(), recommend_result.rating.tolist(),
movielens.test_user2items, recommend_result.user2items, k=10)
print(metrics)
# 因子数kと精度の関係
for factors in [5, 10, 30]:
    recommend_result = recommender.recommend(movielens, factors=factors, fillna_with_zero=False)
    metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
    print(metrics)
