# 非負値行列分解
# 親のフォルダのパスを追加
import sys; sys.path.insert(0, '..')

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
# Movielensのデータの読み込み
data_loader = DataLoader(num_users=1000, num_test_items=5, data_path='../data/ml-10M100K/')
movielens = data_loader.load()
# NMFレコメンド
from src.nmf import NMFRecommender
recommender = NMFRecommender()
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
