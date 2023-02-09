"""カテゴリ変数のカラムをcategory型に変換します"""
from gbdt_wrap.data_loader.data_processor_base import DataProcessorBase
from gbdt_wrap.data_loader.target_data import TargetData


class ToCategoryProcessor(DataProcessorBase):
    """カテゴリ変数のカラムをcategory型に変換します"""

    def process(self, data: TargetData):
        for cat in data.categories:
            data.exp_val[cat] = \
                data.exp_val[cat].astype('category')
