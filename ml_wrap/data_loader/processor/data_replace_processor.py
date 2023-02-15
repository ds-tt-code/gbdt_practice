"""指定の列の値をルールにしたがって置換します"""
from pandas import DataFrame

from ml_wrap.data_loader.processor.data_processor_base import DataProcessorBase


class DataReplaceProcessor(DataProcessorBase):
    """指定の列の値をルールにしたがって置換します"""

    def __init__(self, target_col, replace_dict, after_type=float):
        """コンストラクタです

        Args:
            target_col (str): 置換対象の列名
            replace_dict (dict[str, float]): 置換したい文字と、オブジェクトのdict
        """
        self.target_col = target_col
        self.replace_dict = replace_dict
        self.after_type = after_type

    def _process(self, data: DataFrame) -> DataFrame:
        ret = data.copy()
        ret[self.target_col] = (data[self.target_col]
                                .map(self.replace)
                                .astype(self.after_type))
        return ret

    def replace(self, val):
        """文字列を置換します"""
        ret = str(val)
        for key, val in self.replace_dict.items():
            ret = ret.replace(key, val)
        return ret
