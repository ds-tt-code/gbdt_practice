"""LightGBMのラップクラスです"""
import lightgbm as lgb
from lightgbm import Booster
from pandas import DataFrame

from gbdt_wrap.data_loader.data_loader_base import DataLoaderBase
from gbdt_wrap.gbdt_wrap.data_def import EvalMetric, Objective
from gbdt_wrap.gbdt_wrap.gbdt_base import GBDTBase


class LightGBMWrap(GBDTBase):

    _OBJECTIVE_PARAM = {
        Objective.binary: 'binary'
    }

    _EVAL_METRIC_PARAM = {
        EvalMetric.logloss: 'binary_logloss'
    }

    def __init__(self,
                 loader: DataLoaderBase,
                 objective: Objective = Objective.binary,
                 eval_metric: EvalMetric = EvalMetric.logloss,
                 max_depth=4,
                 seed=None):
        """コンストラクタです

        Args:
            loader (DataLoaderBase): データのローディング処理です。
            objective (str, optional): _description_. Defaults to 'binary:logistic'.
            eval_metric (str, optional): _description_. Defaults to 'logloss'.
            max_depth (int, optional): _description_. Defaults to 4.
            seed (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(loader, seed)

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

    def category_process(self):
        """LightGBM用カテゴリ変数に対する処理"""
        for cat in self.loader.categories:
            self.loader.exp_val[cat] = \
                self.loader.exp_val[cat].astype('category')

    def _trans_data(self,
                    train_exp,
                    train_obj,
                    validation_exp,
                    validation_obj):
        """ LightGBM用データセットの定義

        Args:
            train_exp (_type_): _description_
            train_obj (_type_): _description_
            validation_exp (_type_): _description_
            validation_obj (_type_): _description_

        Returns:
            _type_: _description_
        """
        return lgb.Dataset(train_exp, label=train_obj), \
            lgb.Dataset(validation_exp, label=validation_obj)

    def _get_model(self,
                   train_data: lgb.Dataset,
                   validation_data: lgb.Dataset) -> Booster:
        """モデルを生成します

        Args:
            train_data (lgb.Dataset): 訓練用データ
            validation_data (lgb.Dataset): 検証用データ

        Returns:
            _type_: _description_
        """
        # モデルのfit
        return lgb.train(
            self.params,
            train_set=train_data,
            num_boost_round=self.NROUND,
            valid_sets=[train_data, validation_data],
            valid_names=['training', 'valid'],
            early_stopping_rounds=self.ESR,
            verbose_eval=self.LOGLEVEL
        )

    def _predict(self,
                 model: Booster,
                 validation_exp: DataFrame):
        """Out of Fold法を用いて訓練データの予測を行う

        Args:
            model (Booster): _description_
            validation_exp (DataFrame): _description_

        Returns:
            _type_: _description_
        """
        return model.predict(validation_exp,
                             num_iteration=model.best_iteration)

    def _get_importance(self,
                        model: Booster) -> DataFrame:
        """重要度を計算する

        Args:
            model (Booster): _description_

        Returns:
            _type_: _description_
        """
        importance = DataFrame({
            'feature': list(model.feature_name()),
            'gain': list(model.feature_importance()),
            })

        return importance
