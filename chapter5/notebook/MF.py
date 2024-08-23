# Matrix Factorization
# 親のフォルダのパスを追加
import sys; sys.path.insert(0, '..')

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
# Movielensのデータの読み込み
data_loader = DataLoader(num_users=1000, num_test_items=5, data_path='../data/ml-10M100K/')
movielens = data_loader.load()
# MFレコメンド
from src.mf import MFRecommender
recommender = MFRecommender()
recommend_result = recommender.recommend(movielens)
#  評価
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
print(metrics)
# 評価数のしきい値と精度の関係
for minimum_num_rating in [0, 10, 100, 300]:
    recommend_result = recommender.recommend(movielens, minimum_num_rating=minimum_num_rating)
    metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
    print(metrics)
# 因子数kと精度の関係
for factors in [5, 10, 30]:
    recommend_result = recommender.recommend(movielens, factors=factors)
    metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
    print(metrics)
