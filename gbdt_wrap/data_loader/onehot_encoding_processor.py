"""one hot encodingするデータ処理クラス"""
from sklearn.preprocessing import OneHotEncoder

from gbdt_wrap.data_loader.data_processor_base import DataProcessorBase
from gbdt_wrap.data_loader.target_data import TargetData


class OneHotEncodingProcessor(DataProcessorBase):
    """すべてのカテゴリ変数をone hot encodingするクラスです"""

    def process(self, data: TargetData):
        """すべてのカテゴリ変数に関してone-hot-encodingします

        Args:
            loader (DataLoaderBase): _description_
        """
        # OneHotEncoderのインスタンスを作成
        ohe = OneHotEncoder(sparse=False, categories='auto')

        # カラムに対してOneHotEncoderを適用
        ohe.fit(data.raw_data[data.categories])

        # oheインスタンスからカラム名を作成
        columns = []
        for i, t in enumerate(data.categories):
            columns += [f'{t}_{v}' for v in ohe.categories_[i]]

        # OneHotEncoderに変換
        ohe.transform(data.raw_data[data.categories],
                      columns=columns)
