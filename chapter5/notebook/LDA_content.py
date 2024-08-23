# Latent Dirichlet Allocation (LDA)
# 親のフォルダのパスを追加
import sys; sys.path.insert(0, '..')

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
# Movielensのデータの読み込み
data_loader = DataLoader(num_users=1000, num_test_items=5, data_path='../data/ml-10M100K/')
movielens = data_loader.load()
import gensim
import logging
from gensim.corpora.dictionary import Dictionary

movie_content = movielens.item_content.copy()
# tagが付与されていない映画もあるが、genreはすべての映画に付与されている
# tagとgenreを結合したものを映画のコンテンツ情報として似ている映画を探して推薦していく
# tagがない映画に関しては、NaNになっているので、空のリストに変換してから処理をする
movie_content['tag_genre'] = movie_content['tag'].fillna("").apply(list) + movie_content['genre'].apply(list)
movie_content['tag_genre'] = movie_content['tag_genre'].apply(lambda x:list(map(str, x)))

# タグとジャンルデータを使って、LDAを学習する
tag_genre_data = movie_content.tag_genre.tolist()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
common_dictionary = Dictionary(tag_genre_data)
common_corpus = [common_dictionary.doc2bow(text) for text in tag_genre_data]

# LDAの学習
lda_model = gensim.models.LdaModel(common_corpus, id2word=common_dictionary, num_topics=50, passes=30)
lda_topics = lda_model[common_corpus]



# LDAContentレコメンド
from src.lda_content import LDAContentRecommender
recommender = LDAContentRecommender()
recommend_result = recommender.recommend(movielens)
#  評価
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(
    movielens.test.rating.tolist(), recommend_result.rating.tolist(),
    movielens.test_user2items, recommend_result.user2items, k=10)
print(metrics)
