"""学習対象のデータを表すクラス"""
from pandas import DataFrame
from typing import Tuple
from sklearn.model_selection import StratifiedKFold


class TargetData(object):
    """学習対象のデータを表すクラス"""

    def __init__(self,
                 data: DataFrame,
                 target: str,
                 categories: list[str]):
        """コンストラクタです

        Args:
            data (DataFrame): 対象のデータ
            target(str): 目的変数のカラム名
            categories (list[str]): カテゴリ変数のリスト
        """
        self._raw_data = data
        self._target = target
        self.categories = categories
        self.model = None  # 学習済みもモデル格納用
        self.predict_data = None  # 予測値格納用

        self.initialize_data()

    def initialize_data(self):
        """データを初期化します"""
        self.exp_val, self.obj_val = \
            self._divide_variables(self._raw_data)

    def _divide_variables(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """データセットを目的変数と説明変数に分割します"""
        feature = [col
                   for col
                   in df.columns
                   if col not in self._target]

        return df[feature], df[self._target]  # 目的変数

    def get_nfold_indexes(self,
                          n_fold: int,
                          seed: int = None,
                          ) -> Tuple[list, list]:
        """与えられたデータを訓練用データと検証データに分割します

        Args:
            X_train (DataFrame): 説明変数
            y_train (DataFrame): 目的変数
            n_fold (int): fold数

        Returns:
            Tuple[list, list]: [訓練用データインデックスのリスト,
                                検証用のインデックスのリスト]
        """
        self.folds = StratifiedKFold(
            n_splits=n_fold,
            shuffle=True
        )
        if seed:
            self.folds.random_state = seed
        return self.folds.split(self.exp_val.values,
                                self.obj_val.values)

    def get_train_data(self, train_index: list) -> Tuple[DataFrame, DataFrame]:
        """指定フォールドの訓練用データを取得します
        Args:
            train_index (list): インデックス

        Returns:
            train_exp, train_obj (DataFrame, DataFrame): 説明変数, 目的変数
        """

        train_exp = self.exp_val.iloc[train_index]
        train_obj = self.obj_val.iloc[train_index]
        return train_exp, train_obj

    def get_validation_data(self, validation_index: list) -> Tuple[DataFrame, DataFrame]:
        """指定フォールドの検証用データを取得します
        Args:
            validation_index (list): インデックス

        Returns:
            validation_exp , validation_obj (DataFrame, DataFrame): 説明変数, 目的変数
        """

        validation_exp = self.exp_val.iloc[validation_index]
        validation_obj = self.obj_val.iloc[validation_index]
        return validation_exp, validation_obj
