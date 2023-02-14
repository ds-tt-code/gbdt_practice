"""同じファイル構造のCSVデータを読み込むクラス"""
from glob import glob
from pandas import DataFrame, concat, read_csv

from gbdt_wrap.data_loader.loader.data_loader_base import DataLoaderBase
from gbdt_wrap.data_loader.processor.data_processor_base import DataProcessorBase


class SameCSVLoader(DataLoaderBase):
    """同じ構造のCSVファイルをロードします"""

    def __init__(self,
                 dir: str,
                 target: str = None,
                 categories: list[str] = None,
                 processor: DataProcessorBase = None):
        """コンストラクタです

        Args:
            dir (str): 対象ディレクトリ
            target (_type_): _description_
            categories (_type_): _description_
            processor (_type_): _description_
        """

        self.dir = dir
        super().__init__(target,
                         categories,
                         None,
                         processor=processor)

    def _load(self) -> DataFrame:
        """データをロードします"""
        files = glob(f'{self.dir}/*.csv')
        df = concat([read_csv(f)
                     for f in files])
        return df
