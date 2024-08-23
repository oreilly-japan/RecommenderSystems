"""
# アソシエーション分析
"""

# 親のフォルダのパスを追加
import sys; sys.path.insert(0, '..')

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
# Movielensのデータの読み込み
data_loader = DataLoader(num_users=1000, num_test_items=5, data_path='../data/ml-10M100K/')
movielens = data_loader.load()
# ユーザー×映画の行列形式に変更
user_movie_matrix = movielens.train.pivot(index='user_id', columns='movie_id', values='rating')

# ライブラリ使用のために、4以上の評価値は1, 4未満の評価値と欠損値は0にする
user_movie_matrix[user_movie_matrix < 4] = 0
user_movie_matrix[user_movie_matrix.isnull()] = 0
user_movie_matrix[user_movie_matrix >= 4] = 1

user_movie_matrix.head()
from mlxtend.frequent_patterns import apriori

# 支持度が高い映画の表示
freq_movies = apriori(
    user_movie_matrix, min_support=0.1, use_colnames=True)
freq_movies.sort_values('support', ascending=False).head()
# movie_id=593のタイトルの確認
movielens.item_content[movielens.item_content.movie_id == 593]
from mlxtend.frequent_patterns import association_rules

# アソシエーションルールの計算（リフト値の高い順に表示）
rules = association_rules(freq_movies, metric='lift', min_threshold=1)
rules.sort_values('lift', ascending=False).head()[['antecedents', 'consequents', 'lift']]
# Associationレコメンド
from src.association import AssociationRecommender
recommender = AssociationRecommender()
recommend_result = recommender.recommend(movielens)
#  評価
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
print(metrics)
# min_supportと精度の関係
for min_support in [0.06, 0.07, 0.08, 0.09, 0.1, 0.11]:
    recommend_result = recommender.recommend(movielens, min_support=min_support)
    metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
    print(metrics)
