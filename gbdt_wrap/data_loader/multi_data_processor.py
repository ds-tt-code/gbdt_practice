"""複数のデータ処理を実行するクラス"""
from gbdt_wrap.data_loader import DataProcessorBase, DataLoaderBase


class MultiDataProcessor(DataProcessorBase):

    def __init__(self):
        """コンストラクタ"""
        self.processor_list: list[DataProcessorBase] = []

    def add_processor(self, processor: DataProcessorBase):
        """データ処理を追加します

        Args:
            processor (DataProcessorBase): データ処理クラス
        """
        self.processor_list.append(processor)

    def process(self, loader: DataLoaderBase):
        for p in self.processor_list:
            p.process(loader)
