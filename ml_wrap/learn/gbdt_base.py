"""GBDTのベースクラス"""
from abc import abstractmethod

from ml_wrap.learn.model_wrap_base import ModelWrapBase, evaluate_index
from ml_wrap.data_loader.processor.data_processor_base import DataProcessorBase


class GBDTBase(ModelWrapBase):
    """GBDTのベースクラスです"""
    N_ROUND = 100000  # 基本的は大きい値で固定。これで調整することはない
    LOGLEVEL = 5

    def __init__(self,
                 data_processor: DataProcessorBase,
                 result_label: str = None,
                 early_stopping_rounds: int = 20,
                 seed: int = None):
        """コンストラクタ

        Args:
            loader (DataLoaderBase): _description_
            seed (int, optional): _description_. Defaults to None.
        """

        super().__init__(data_processor, result_label)

        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds

    @evaluate_index('importance')
    def importance(self):
        return [self.calc_importance(r.model)
                for r
                in self.result]

    @abstractmethod
    def calc_importance(self, model):
        pass
