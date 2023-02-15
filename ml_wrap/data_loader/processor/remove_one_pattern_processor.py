"""データ種別が1種類しかない列を削除します"""
from pandas import DataFrame

from ml_wrap.data_loader.processor.data_processor_base import DataProcessorBase


class RemoveOnePatternProcessor(DataProcessorBase):
    """データの種類が一種類しかない列を削除します"""

    def _process(self, data: DataFrame) -> DataFrame:
        one_pattern_list = [col
                            for col, cnt
                            in data.nunique().items()
                            if cnt == 1]

        return data.drop(one_pattern_list, axis=1)
