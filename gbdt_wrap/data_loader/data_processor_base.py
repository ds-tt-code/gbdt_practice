"""データ前処理のベースクラス"""
from abc import ABCMeta, abstractmethod

from gbdt_wrap.data_loader.target_data import TargetData


class DataProcessorBase(object, metaclass=ABCMeta):

    @abstractmethod
    def process(self, loader: TargetData):
        """データを加工します

        Args:
            loader (DataLoaderBase): データローダークラス
        """
        pass
