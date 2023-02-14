"""指定されたの列を削除します"""
from pandas import DataFrame

from gbdt_wrap.data_loader.processor.data_processor_base import DataProcessorBase


class DataDropProcessor(DataProcessorBase):
    """すべてnullの列を削除します"""

    def __init__(self, *cols):
        """コンストラクタ"""
        self.cols = list(cols)

    def _process(self, data: DataFrame) -> DataFrame:
        return data.drop(self.cols, axis=1)
