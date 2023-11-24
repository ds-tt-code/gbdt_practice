"""ホールドアウト法でデータを分割するクラスです"""
from pandas import DataFrame
from numpy import array
from sklearn.model_selection import train_test_split

from ml_wrap.learn.divider.data_divider_base import DataDividerBase


class HoldOutDivider(DataDividerBase):
    """ホールドアウト法でデータを分割するクラス"""

    def __init__(self, seed=None):
        super().__init__(seed)

    def divide_train_validation(self,
                                exp_vals: DataFrame,
                                obj_vals: DataFrame):
        """分割を実行し、インデックスを返します

        Args:
            exp_vals (DataFrame): 説明変数
            obj_vals (DataFrame): 目的変数
        """

        return [
                tuple(
                    map(
                        array,
                        train_test_split(list(range(len(exp_vals))),
                                         random_state=self.seed)
                        )
                      ),
                ]
