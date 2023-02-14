"""指定された列のNaを指定文字列でパディングします"""
from pandas import DataFrame

from gbdt_wrap.data_loader.processor.data_processor_base import DataProcessorBase


class NaPaddingProcessor(DataProcessorBase):
    """指定された列のNaを指定文字列でパディングします
        パディングするときは文字列をパディングするので対象列をobject型に変換します
    """

    def __init__(self, target: list[str], padding_str: str):
        """コンストラクタです

        Args:
            target (list[str]): パディングする列
            padding_str (str): パディングする文字列
        """
        self.target = target
        self.padding_str = padding_str

    def _process(self, data: DataFrame) -> DataFrame:
        ret = data.copy()

        for t in self.target:
            ret[t] = data[t].astype(object)
            ret[t].fillna(self.padding_str, inplace=True)

        return ret
