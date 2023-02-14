"""カテゴリ変数を処理するクラスのベースクラス"""
from gbdt_wrap.data_loader.processor import DataProcessorBase
from gbdt_wrap.data_loader.target_data import TargetData


class CategoryProcessorBase(DataProcessorBase):
    """カテゴリ変数を処理するクラスのベース"""

    def process(self, data: TargetData) -> TargetData:
        """データを加工します

        Args:
            loader (DataLoaderBase): データローダークラス
        """
        self.categories = data.categories
        return super().process(data)
