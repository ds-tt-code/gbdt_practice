"""データ読み込みのベースクラス"""
from pandas import DataFrame

from gbdt_wrap.data_loader.target_data import TargetData
from gbdt_wrap.data_loader.processor.data_processor_base import DataProcessorBase


class DataLoaderBase():
    """データローディングと加工を行うクラスです"""

    def __init__(self,
                 loader: callable,
                 target: str = None,
                 categories: list[str] = None,
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
           データ処理が指定されていた場合はデータ処理も行います

        Raises:
            NotImplementedError: 子クラス側で実装されていない
                                 または読み込み関数が指定されていない
                                 場合はエラーを発出

        Returns:
            DataFrame: 読み込んだデータを返します
        """
        data = self._load()

        if self.loader:
            data = self.loader()

        if len(data) == 0:
            raise NotImplementedError('継承して読み込み処理を実装するか読み込み関数を指定してください')

        ret = self.df_to_target_data(data)

        if self.processor:
            self.processor.process(ret)
        return ret

    def _load(self) -> DataFrame:
        return DataFrame()

    def df_to_target_data(self, data: DataFrame) -> TargetData:
        """pandasのdataframeをtargetdataに変換します

        Args:
            data (DataFrame): 読み込んだdataframe

        Returns:
            TargetData: _description_
        """
        return TargetData(data,
                          self.target,
                          self.categories)
