"""すべてnullの列を削除します"""
from pandas import DataFrame

from ml_wrap.data_loader.processor.data_processor_base import DataProcessorBase


class RemoveAllNullProcessor(DataProcessorBase):
    """すべてnullの列を削除します"""

    def _process(self, data: DataFrame) -> DataFrame:
        all_null_list = [col
                         for col
                         in data.columns
                         if data[col].count() == 0]

        return data.drop(all_null_list, axis=1)
