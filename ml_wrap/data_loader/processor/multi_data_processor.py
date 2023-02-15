"""複数のデータ処理を実行するクラス"""
from pandas import DataFrame

from ml_wrap.data_loader.processor.data_processor_base import DataProcessorBase
from ml_wrap.data_loader.target_data import TargetData


class MultiDataProcessor(DataProcessorBase):

    def __init__(self, *processor):
        """コンストラクタ"""
        self.processor_list: list[DataProcessorBase] = processor

    def add_processor(self, processor: DataProcessorBase):
        """データ処理を追加します

        Args:
            processor (DataProcessorBase): データ処理クラス
        """
        self.processor_list.append(processor)

    def process(self, data: TargetData) -> TargetData:
        ret = data
        for p in self.processor_list:
            ret = p.process(ret)
        return ret

    def _process(self, data: DataFrame) -> DataFrame:
        return None
