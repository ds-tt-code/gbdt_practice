"""one hot encodingするデータ処理クラス"""
from sklearn.preprocessing import OneHotEncoder
from pandas import DataFrame

from gbdt_wrap.data_loader.processor.category_processor_base import CategoryProcessorBase


class OneHotEncodingProcessor(CategoryProcessorBase):
    """すべてのカテゴリ変数をone hot encodingするクラスです"""

    def _process(self, data: DataFrame) -> DataFrame:
        """すべてのカテゴリ変数に関してone-hot-encodingします

        Args:
            loader (DataLoaderBase): _description_
        """
        # OneHotEncoderのインスタンスを作成
        ohe = OneHotEncoder(sparse=False, categories='auto')

        # カラムに対してOneHotEncoderを適用
        ohe.fit(data[self.categories])

        # oheインスタンスからカラム名を作成
        columns = []
        for i, t in enumerate(self.categories):
            columns += [f'{t}_{v}' for v in ohe.categories_[i]]

        # OneHotEncoderに変換
        ret = data.copy()
        ohe.transform(ret[self.categories],
                      columns=columns)

        return ret
