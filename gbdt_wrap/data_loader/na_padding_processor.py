"""指定された列のNaを指定文字列でパディングします"""
from gbdt_wrap.data_loader.data_processor_base import DataProcessorBase
from gbdt_wrap.data_loader.target_data import TargetData


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

    def process(self, data: TargetData):
        print(data.exp_val)
        print(self.padding_str)

        for t in self.target:
            data.exp_val[t] = data.exp_val[t].astype(object)
            data.exp_val[t].fillna(self.padding_str, inplace=True)

        print(data.exp_val)
