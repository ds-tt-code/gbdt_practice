from sklearn.preprocessing import LabelEncoder

from gbdt_wrap.data_loader.data_processor_base import DataProcessorBase
from gbdt_wrap.data_loader.target_data import TargetData


class LabelEncoderProcessor(DataProcessorBase):
    """カテゴリ変数をすべてLabelEncodingするクラスです"""

    def process(self, data: TargetData):
        le = LabelEncoder()
        for cat in data.categories:
            data.exp_val[cat] = le.fit_transform(
                                    data.exp_val[cat]
                                )
