
"""カテゴリ変数のNaを指定文字列でパディングします"""
from pandas import DataFrame

from ml_wrap.data_loader.processor.category_processor_base import CategoryProcessorBase


class NaPaddingProcessor(CategoryProcessorBase):
    """指定された列のNaを指定文字列でパディングします
        パディングするときは文字列をパディングするので対象列をobject型に変換します
    """

    def __init__(self, padding_str: str):
        """コンストラクタです

        Args:
            target (list[str]): パディングする列
            padding_str (str): パディングする文字列
        """
        self.padding_str = padding_str

    def _process(self, data: DataFrame) -> DataFrame:
        ret = data.copy()

        for t in self.categories:
            ret[t] = data[t].astype(object)
            ret[t].fillna(self.padding_str, inplace=True)

        return ret
