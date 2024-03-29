"""カテゴリ変数のカラムをcategory型に変換します"""
from pandas import DataFrame

from ml_wrap.data_loader.processor.category_processor_base import CategoryProcessorBase


class ToCategoryProcessor(CategoryProcessorBase):
    """カテゴリ変数のカラムをcategory型に変換します"""

    def _process(self, data: DataFrame) -> DataFrame:
        ret = data.copy()
        for cat in self.categories:
            ret[cat] = \
                data[cat].astype('category')
        return ret
