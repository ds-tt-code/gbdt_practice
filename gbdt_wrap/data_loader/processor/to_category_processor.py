"""カテゴリ変数のカラムをcategory型に変換します"""
from pandas import DataFrame

from gbdt_wrap.data_loader.processor.category_processor_base import CategoryProcessorBase


class ToCategoryProcessor(CategoryProcessorBase):
    """カテゴリ変数のカラムをcategory型に変換します"""

    def _process(self, data: DataFrame) -> DataFrame:
        for cat in self.categories:
            data[cat] = \
                data[cat].astype('category')
