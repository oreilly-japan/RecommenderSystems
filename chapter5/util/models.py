import dataclasses
import pandas as pd
from typing import Dict, List


@dataclasses.dataclass(frozen=True)
# 推薦システムの学習と評価に使うデータセット
class Dataset:
    # 学習用の評価値データセット
    train: pd.DataFrame
    # テスト用の評価値データセット
    test: pd.DataFrame
    # ランキング指標のテストデータセット。キーはユーザーID、バリューはユーザーが高評価したアイテムIDのリスト。
    test_user2items: Dict[int, List[int]]
    # アイテムのコンテンツ情報
    item_content: pd.DataFrame


@dataclasses.dataclass(frozen=True)
# 推薦システムの予測結果
class RecommendResult:
    # テストデータセットの予測評価値。RMSEの評価
    rating: pd.DataFrame
    # キーはユーザーID、バリューはおすすめアイテムIDのリスト。ランキング指標の評価。
    user2items: Dict[int, List[int]]


@dataclasses.dataclass(frozen=True)
# 推薦システムの評価
class Metrics:
    rmse: float
    precision_at_k: float
    recall_at_k: float

    # 評価結果を出力する時に少数は第３桁までにする
    def __repr__(self):
        return f"rmse={self.rmse:.3f}, Precision@K={self.precision_at_k:.3f}, Recall@K={self.recall_at_k:.3f}"
