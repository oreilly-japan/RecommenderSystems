"""
# 人気度順推薦
## 人気度の定義
* 今回は評価値が高いものを人気が髙い映画とする
* 人気度の定義は「クリック数が多いもの」「購入が多いもの」「評価値が髙いもの」など複数あり、自社サービスに最も適した定義を利用
"""

# 親のフォルダのパスを追加
import sys; sys.path.insert(0, '..')

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
# Movielensのデータの読み込み
data_loader = DataLoader(num_users=1000, num_test_items=5, data_path='../data/ml-10M100K/')
movielens = data_loader.load()

import numpy as np
# 評価値が髙い映画の確認
movie_stats = movielens.train.groupby(['movie_id', 'title']).agg({'rating': [np.size, np.mean]})
movie_stats.sort_values(by=('rating', 'mean'), ascending=False).head()
# しきい値を導入
movie_stats = movielens.train.groupby(['movie_id', 'title']).agg({'rating': [np.size, np.mean]})
atleast_flg = movie_stats['rating']['size'] >= 100
movies_sorted_by_rating = movie_stats[atleast_flg].sort_values(by=('rating', 'mean'), ascending=False)
movies_sorted_by_rating.head()
# 人気度推薦
from src.popularity import PopularityRecommender
recommender = PopularityRecommender()
recommend_result = recommender.recommend(movielens, minimum_num_rating=100)
#  評価
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
print(metrics)
# しきい値を変更したときの挙動
for minimum_num_rating in [1, 200]:
    recommend_result = recommender.recommend(movielens, minimum_num_rating=minimum_num_rating)
    metrics = metric_calculator.calc(
        movielens.test.rating.tolist(), recommend_result.rating.tolist(),
        movielens.test_user2items, recommend_result.user2items, k=10)
    print(metrics)