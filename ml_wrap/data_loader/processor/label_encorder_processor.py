from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame

from ml_wrap.data_loader.processor.category_processor_base import CategoryProcessorBase


class LabelEncoderProcessor(CategoryProcessorBase):
    """カテゴリ変数をすべてLabelEncodingするクラスです"""

    def _process(self, data: DataFrame) -> DataFrame:
        le = LabelEncoder()
        ret = data.copy()
        for cat in self.categories:
            ret[cat] = le.fit_transform(
                                    data[cat]
                       )
