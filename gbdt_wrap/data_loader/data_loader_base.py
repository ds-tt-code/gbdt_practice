"""データ読み込みのベースクラス"""
from typing import Tuple

from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold


class DataLoaderBase():
    """データローディングと加工を行うクラスです"""

    def __init__(self,
                 target: str,
                 categories: list[str],
                 loader: callable = None):
        """コンストラクタ

        Args:
            seed (int): データ分割のランダムシード値
            target(str): 目的変数のカラム名
            categories (list[str]): カテゴリ変数のリスト
            loader (callable): ロード関数
        """
        self.target = target
        self.categories = categories
        self.loader = loader

        self.initialize_data()

    def load(self) -> DataFrame:
        """データをロードします

        Raises:
            NotImplementedError: 子クラス側で実装されていない
                                 または読み込み関数が指定されていない
                                 場合はエラーを発出

        Returns:
            DataFrame: 読み込んだデータを返します
        """
        if self.loader:
            return self.loader()

        raise NotImplementedError('継承して読み込み処理を実装するか読み込み関数を指定してください')

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

    def _divide_variables(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """データセットを目的変数と説明変数に分割します"""
        feature = [col
                   for col
                   in df.columns
                   if col not in self.target]

        return df[feature], df[self.target]  # 目的変数

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

    def initialize_data(self):
        """データを初期化します"""
        self.raw_data = self.load()
        self.exp_val, self.obj_val = self._divide_variables(self.raw_data)
