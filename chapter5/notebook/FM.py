# Factorization Machiens
# 親のフォルダのパスを追加
import sys; sys.path.insert(0, '..')

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
# Movielensのデータの読み込み
data_loader = DataLoader(num_users=1000, num_test_items=5, data_path='../data/ml-10M100K/')
movielens = data_loader.load()
# FMレコメンド
from src.fm import FMRecommender
recommender = FMRecommender()
recommend_result = recommender.recommend(movielens)
#  評価
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
print(metrics)
# 補助情報の利用
recommend_result = recommender.recommend(movielens, use_side_information=True)
metrics = metric_calculator.calc(
movielens.test.rating.tolist(), recommend_result.rating.tolist(),
movielens.test_user2items, recommend_result.user2items, k=10)
print(metrics)

