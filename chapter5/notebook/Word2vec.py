# Word2vec
# 親のフォルダのパスを追加
import sys; sys.path.insert(0, '..')

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
# Movielensのデータの読み込み
data_loader = DataLoader(num_users=1000, num_test_items=5, data_path='../data/ml-10M100K/')
movielens = data_loader.load()
import gensim
import logging

movie_content = movielens.item_content.copy()

# tagが付与されていない映画もあるが、genreはすべての映画に付与されている
# tagとgenreを結合したものを映画のコンテンツ情報として似ている映画を探して推薦していく
# tagがない映画に関しては、NaNになっているので、空のリストに変換してから処理をする
movie_content['tag_genre'] = movie_content['tag'].fillna("").apply(list) + movie_content['genre'].apply(list)
movie_content['tag_genre'] = movie_content['tag_genre'].apply(lambda x:set(map(str, x)))

# タグとジャンルデータを使って、word2vecを学習する
tag_genre_data = movie_content.tag_genre.tolist()
model = gensim.models.word2vec.Word2Vec(tag_genre_data, vector_size=100, window=100, sg=1, hs=0, epochs=50, min_count=5)

# animeタグに似ているタグを確認
model.wv.most_similar('anime')
# Word2vecContentレコメンド
from src.word2vec import Word2vecRecommender
recommender = Word2vecRecommender()
recommend_result = recommender.recommend(movielens)
#  評価
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
print(metrics)

