"""k-fold cross validationでデータを分割するクラスです"""
from pandas import DataFrame
from sklearn.model_selection import KFold

from ml_wrap.learn.divider.data_divider_base import DataDividerBase


class KFoldCrossValidationDivider(DataDividerBase):
    """k-fold cross validationでデータを分割するクラス"""

    def __init__(self, n_splits=5, seed=None):
        super().__init__(seed)
        self.n_splits = n_splits

    def divide_train_validation(self,
                                exp_vals: DataFrame,
                                obj_vals: DataFrame):
        """分割を実行し、インデックスを返します

        Args:
            exp_vals (DataFrame): 説明変数
            obj_vals (DataFrame): 目的変数
        """
        kf = KFold(n_splits=self.n_splits,
                   shuffle=True,
                   random_state=self.seed)

        return kf.split(exp_vals, obj_vals)
