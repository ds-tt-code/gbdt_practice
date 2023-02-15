"""データ前処理のベースクラス"""
from abc import ABCMeta, abstractmethod
from pandas import DataFrame

from ml_wrap.data_loader.target_data import TargetData


class DataProcessorBase(object, metaclass=ABCMeta):

    def process(self, data: TargetData) -> TargetData:
        """データを加工します

        Args:
            loader (DataLoaderBase): データローダークラス
        """
        ret = self._process(data._raw_data)
        return TargetData(ret, data.target, data.categories)

    @abstractmethod
    def _process(self, data: DataFrame) -> DataFrame:
        pass
