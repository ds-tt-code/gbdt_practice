from abc import ABCMeta, abstractmethod

from pandas import DataFrame
from sklearn.metrics import log_loss
from ml_wrap.data_loader.processor.data_processor_base import DataProcessorBase
from ml_wrap.data_loader import TargetData
from ml_wrap.learn.divider.data_divider_base import DataDividerBase
from ml_wrap.learn.ml_result_wrap import MlResultWrap


class ModelWrapMeta(ABCMeta):
    """GBDTのメタクラスです。
        learnメソッドの前にpre_data_processが呼ばれることを強制し、
        評価メソッドを集計し、evaluatesというクラス変数に関数一覧をリスト形式で格納します
    """

    def __new__(meta, name, bases, attributes):
        cls = super().__new__(meta, name, bases, attributes)
        if name != 'ModelWrapBase':
            p_cls = cls
            cls.evaluates = []
            while p_cls != object:
                cls.evaluates.extend([
                    func for func in p_cls.__dict__.values() if hasattr(func, '_evaluate_name')
                ])
                p_cls = p_cls.__base__
        return cls

    @classmethod
    def add_process_func(cls, learn_func):
        """データ処理を学習の前に入れます"""
        def new_learn(self, data: TargetData, divider: DataDividerBase):
            data = ModelWrapBase.pre_data_process(self, data)
            learn_func(self, data, divider)
        return new_learn


def evaluate_index(name):
    """評価指標に付与するデコレータです"""

    def add_valuate_index(func):
        func._evaluate_name = name
        return func

    return add_valuate_index


class ModelWrapBase(object, metaclass=ModelWrapMeta):
    """モデルラップのベースクラス"""

    def __init__(self,
                 data_processor: DataProcessorBase = None,
                 result_label: str = None):
        self.data_processor = data_processor
        self.model = None
        self.result_label = result_label or self.__class__.__name__

    def learn(self,
              data: TargetData,
              divider: DataDividerBase):
        """学習してモデルを生成します

        Args:
        """
        data = self.pre_data_process(data)

        self.result: MlResultWrap = []
        for fold, (train_exp,
                   train_obj,
                   validation_exp,
                   validation_obj) in enumerate(divider.divide(data)):
            model = self._learn(train_exp,
                                train_obj,
                                validation_exp,
                                validation_obj)

            label = f'fold_{fold}_{self.result_label}'
            self.result.append(
                MlResultWrap(
                    validation_exp,
                    validation_obj,
                    model,
                    label
                )
            )

    @abstractmethod
    def _learn(self,
               train_exp: DataFrame,
               train_obj: DataFrame,
               validation_exp: DataFrame,
               validation_obj: DataFrame):
        """学習を実行します

        Args:
            train_exp (_type_): _description_
            train_obj (_type_): _description_
            validation_exp (_type_): _description_
            validation_obj (_type_): _description_

        Returns:
            _type_: importance, loglossの平均, preds(予測)
        """
        pass

    def predict(self, data: DataFrame):
        """予測値を返します"""
        return [self._predict(r.model, data) for r in self.result]

    @abstractmethod
    def _predict(self,
                 model,
                 validation_exp: DataFrame):
        """Out of Fold法を用いて訓練データの予測を行う

        Args:
            model (Booster): _description_
            validation_exp (DataFrame): _description_

        Returns:
            _type_: _description_
        """
        pass

    def pre_data_process(self, data: TargetData):
        """データを前処理します"""
        ret = data
        if self.data_processor:
            ret = self.data_processor.process(data)
        return ret

    def get_evaluates(self) -> dict:
        """評価指標の一覧を取得します"""
        return {e._evaluate_name: e(self) for e in self.evaluates}

    @evaluate_index(name='log_loss')
    def log_loss(self):
        """評価指標のlog_lossを取得します

        Returns:
            _type_: _description_
        """
        return [self.calc_log_loss(r.model,
                                   r.validation_exp,
                                   r.validation_obj)
                for r
                in self.result]

    def calc_log_loss(self, model, validation_exp, validation_obj):
        """_summary_

        Args:
            validation_exp (_type_): _description_
            validation_obj (_type_): _description_

        Returns:
            _type_: _description_
        """
        print(validation_obj)
        preds = self._predict(model, validation_exp)
        print(preds)
        print(preds[0])
        return log_loss(validation_obj, preds)
