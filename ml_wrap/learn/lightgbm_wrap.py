"""LightGBMのラップクラスです"""
from lightgbm import Booster, train, Dataset
from pandas import DataFrame

from ml_wrap.data_loader.processor.to_category_processor import ToCategoryProcessor
from ml_wrap.learn.data_def import EvalMetric, Objective
from ml_wrap.learn.gbdt_base import GBDTBase


class LightGBMWrap(GBDTBase):

    _OBJECTIVE_PARAM = {
        Objective.binary: 'binary',
        Objective.regression: 'regression',
        Objective.multi_class: 'multiclass'
    }

    _EVAL_METRIC_PARAM = {
        EvalMetric.logloss: 'binary_logloss',
        EvalMetric.mae: 'mae',

    }

    def __init__(self,
                 objective: Objective,
                 eval_metric: EvalMetric,
                 result_label: str = None,
                 max_depth=4,
                 seed=None,
                 early_stopping_rounds: int = 20):
        """コンストラクタです

        Args:
            loader (DataLoaderBase): データのローディング処理です。
            objective (str, optional): _description_. Defaults to 'binary:logistic'.
            eval_metric (str, optional): _description_. Defaults to 'logloss'.
            max_depth (int, optional): _description_. Defaults to 4.
            seed (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(ToCategoryProcessor(),
                         result_label,
                         early_stopping_rounds=early_stopping_rounds)

        self.params = {
            'objective': self._OBJECTIVE_PARAM[objective],
            'eval_metric': self._EVAL_METRIC_PARAM[eval_metric],
            'max_depth': max_depth,
            'learning_rate': 0.01,
            'num_leaves': 32,
            'verbosity': -1,
        }

        if seed:
            self.params['random_state'] = seed

    def _learn(self,
               train_exp,
               train_obj,
               validation_exp,
               validation_obj):
        """モデルを生成します

        Args:
            train_exp (_type_): _description_
            train_obj (_type_): _description_
            validation_exp (_type_): _description_
            validation_obj (_type_): _description_

        Returns:
            _type_: _description_
        """
        train_exp_ds, validation_exp_ds = \
            Dataset(train_exp, label=train_obj), \
            Dataset(validation_exp, label=validation_obj)

        return train(
            self.params,
            train_set=train_exp_ds,
            num_boost_round=self.N_ROUND,
            valid_sets=[train_exp_ds, validation_exp_ds],
            valid_names=['training', 'valid'],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.LOGLEVEL
        )

    def _predict(self, model, validation_exp: DataFrame):
        """Out of Fold法を用いて訓練データの予測を行う

        Args:
            model (Booster): _description_
            validation_exp (DataFrame): _description_

        Returns:
            _type_: _description_
        """
        return model.predict(validation_exp,
                             num_iteration=model.best_iteration)

    def calc_importance(self, model: Booster):
        """重要度を計算します"""
        return {
            'feature': list(model.feature_name()),
            'gain': list(model.feature_importance()),
        }
