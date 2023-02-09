"""GBDTを実行するクラス"""
from seaborn import barplot

from gbdt_wrap.gbdt_wrap.gbdt_base import GBDTBase


class GBDTExecutor(object):
    """GBDTを簡易に扱うためのクラスです"""

    def __init__(self):
        """コンストラクタ"""
        self.gbdt_obj_list: list[GBDTBase] = []

    def add_target(self, target: GBDTBase):
        """実行するGBDTオブジェクトを追加"""
        self.gbdt_obj_list.append(target)

    def learn_all(self):
        """すべての学習を実行する"""
        for obj in self.gbdt_obj_list:
            importance, logloss, preds = \
                obj.learn_cv(5)
            print(f'execute in {obj.__class__.__name__}==============')
            print(logloss)
            print(f'val_log_loss_mean: {logloss: 4f}')
            barplot(x='gain',
                    y='feature',
                    data=importance.sort_values('gain', ascending=False))
