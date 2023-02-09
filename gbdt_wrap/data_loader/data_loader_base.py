"""データ読み込みのベースクラス"""
from gbdt_wrap.data_loader.target_data import TargetData
from gbdt_wrap.data_loader.data_processor_base import DataProcessorBase


class DataLoaderBase():
    """データローディングと加工を行うクラスです"""

    def __init__(self,
                 target: str,
                 categories: list[str],
                 loader: callable,
                 processor: DataProcessorBase = None):
        """コンストラクタ

        Args:
            target(str): 目的変数のカラム名
            categories (list[str]): カテゴリ変数のリスト
            loader (callable): ロード関数。引数なし、戻り値はDataFrameとしてください, 継承して使用する場合はNoneをしてください
            processor (DataProcessorBase): データ処理クラス。共通のデータ処理がある場合はここに指定。
        """
        self.target = target
        self.categories = categories
        self.loader = loader
        self.processor = processor

    def load(self) -> TargetData:
        """データをロードします

        Raises:
            NotImplementedError: 子クラス側で実装されていない
                                 または読み込み関数が指定されていない
                                 場合はエラーを発出

        Returns:
            DataFrame: 読み込んだデータを返します
        """
        if self.loader:
            raw_data = self.loader()
            return TargetData(raw_data,
                              self.target,
                              self.categories)

        raise NotImplementedError('継承して読み込み処理を実装するか読み込み関数を指定してください')
