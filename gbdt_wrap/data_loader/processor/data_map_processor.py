"""指定された列に指定された関数を適用します"""
from pandas import DataFrame
from gbdt_wrap.data_loader.processor.data_processor_base import DataProcessorBase


class DataMapProcessor(DataProcessorBase):
    """指定された列に指定された関数を適用します"""

    def __init__(self, target_col: str, func: callable, after_type):
        """コンストラクタです

        Args:
            target_col (str): 置換対象の列名
            replace_dict (dict[str, float]): 置換したい文字と、オブジェクトのdict
        """
        self.target_col = target_col
        self.func = func
        self.after_type = after_type

    def _process(self, data: DataFrame) -> DataFrame:
        ret = data.copy()
        ret[self.target_col] = (data[self.target_col]
                                .map(self.func)
                                .astype(self.after_type))
        return ret
