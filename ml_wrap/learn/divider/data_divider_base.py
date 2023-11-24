from abc import ABCMeta, abstractmethod
from typing import Tuple

from pandas import DataFrame

from ml_wrap.data_loader import TargetData


class DataDividerBase(object, metaclass=ABCMeta):
    """データ分割を実行するベースクラス"""

    def __init__(self, seed=None):
        """コンストラクタ

        Args:
            seed (int): シード値
        """
        self.seed = seed

    def divide_exp_obj(self,
                       target: str,
                       data: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """データセットを目的変数と説明変数に分割します。

        Returns:
            train_exp, train_obj (DataFrame, DataFrame): 説明変数, 目的変数
        """
        feature = [col
                   for col
                   in data.columns
                   if col not in target]

        return data[feature], data[target]

    @abstractmethod
    def divide_train_validation(self,
                                exp_vals: DataFrame,
                                obj_vals: DataFrame):
        """訓練データと検証データを分割するためのインデックスを返します
        Args:
            exp_vals (DataFrame): 説明変数
            obj_vals (DataFrame): 目的変数
        """
        pass

    def divide(self, data: TargetData):
        """_summary_

        Args:
            data (TargetData): 学習データ

        Raises:
            ValueError: 学習データに目的変数の設定がされていなければ例外を投げます

        Yields:
            (DataFrame, DataFrame, DataFrame, DataFrame):
                訓練データの説明変数, 訓練データの目的変数, 検証データの説明変数, 検証データの目的変数
        """

        if not data.target:
            raise ValueError('学習データに目的変数の設定がされていません。')

        exp_vals, obj_vals = self.divide_exp_obj(data.target, data._raw_data)
        for train_index, validation_index in self.divide_train_validation(exp_vals, obj_vals):

            train_exp = exp_vals.iloc[train_index]
            train_obj = obj_vals.iloc[train_index]
            validation_exp = exp_vals.iloc[validation_index]
            validation_obj = obj_vals.iloc[validation_index]
            yield train_exp, train_obj, validation_exp, validation_obj
