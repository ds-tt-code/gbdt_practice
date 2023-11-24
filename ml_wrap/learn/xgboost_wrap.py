"""XGBoostのラップクラスです"""
from xgboost import Booster, DMatrix, train
from pandas import DataFrame

from ml_wrap.data_loader.processor.label_encorder_processor import LabelEncoderProcessor
from ml_wrap.learn.data_def import EvalMetric, Objective
from ml_wrap.learn.gbdt_base import GBDTBase


class XGBoostWrap(GBDTBase):

    _OBJECTIVE_PARAM = {
        Objective.binary: 'binary:logistic',
        Objective.regression: 'reg:linear',
        Objective.multi_class: 'multi:softmax'
    }

    _EVAL_METRIC_PARAM = {
        EvalMetric.logloss: 'logloss',
        EvalMetric.mae: 'mae'
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
        super().__init__(LabelEncoderProcessor(),
                         result_label,
                         early_stopping_rounds=early_stopping_rounds)

        self.params = {
            'objective': self._OBJECTIVE_PARAM[objective],
            'eval_metric': self._EVAL_METRIC_PARAM[eval_metric],
            'max_depth': max_depth,
            'verbosity': 0,
            'silent': 1
        }

        if seed:
            self.params['random_state'] = seed

    def _learn(self,
               train_exp,
               train_obj,
               validation_exp,
               validation_obj) -> Booster:
        """モデルを生成します

        Args:
            train_data (xgb.DMatrix): 訓練用データ
            validation_data (xgb.DMatrix): 検証用データ

        Returns:
            _type_: _description_
        """

        train_exp_ds, validation_exp_ds = \
            DMatrix(train_exp, label=train_obj), \
            DMatrix(validation_exp, label=validation_obj)

        return train(
            self.params,
            dtrain=train_exp_ds,
            num_boost_round=self.N_ROUND,
            evals=[(train_exp_ds, 'train'), (validation_exp_ds, 'valid')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.LOGLEVEL
        )

    def _predict(self, model, validation_exp: DataFrame):
        return model.predict(DMatrix(validation_exp),
                             ntree_limit=model.best_ntree_limit)

    def calc_importance(self, model):
        return model.get_score(importance_type='gain')
