"""XGBoostのラップクラスです"""
import xgboost as xgb
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from xgboost import Booster

from gbdt_wrap.data_loader.data_loader_base import DataLoaderBase
from gbdt_wrap.gbdt_wrap.data_def import EvalMetric, Objective
from gbdt_wrap.gbdt_wrap.gbdt_base import GBDTBase


class XGBoostWrap(GBDTBase):

    _OBJECTIVE_PARAM = {
        Objective.binary: 'binary:logistic'
    }

    _EVAL_METRIC_PARAM = {
        EvalMetric.logloss: 'logloss'
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
            'verbosity': 0,
            'silent': 1
        }

        if seed:
            self.params['random_state'] = seed

    def category_process(self):
        """カテゴリ変数処理を実行します"""

        le = LabelEncoder()
        for cat in self.loader.categories:
            self.loader.exp_val[cat] = le.fit_transform(
                                    self.loader.exp_val[cat]
                                )

    def _trans_data(self,
                    train_exp,
                    train_obj,
                    validation_exp,
                    validation_obj):
        """ XGBoost用データセットの定義
            この処理を挟む事で省メモリに学習する事ができる

        Args:
            train_exp (_type_): _description_
            train_obj (_type_): _description_
            validation_exp (_type_): _description_
            validation_obj (_type_): _description_

        Returns:
            _type_: _description_
        """

        return xgb.DMatrix(train_exp, label=train_obj), \
            xgb.DMatrix(validation_exp, label=validation_obj)

    def _get_model(self,
                   train_data: xgb.DMatrix,
                   validation_data: xgb.DMatrix) -> Booster:
        """モデルを生成します

        Args:
            train_data (xgb.DMatrix): 訓練用データ
            validation_data (xgb.DMatrix): 検証用データ

        Returns:
            _type_: _description_
        """

        return xgb.train(
            self.params,
            dtrain=train_data,
            num_boost_round=self.NROUND,
            evals=[(train_data, 'train'), (validation_data, 'valid')],
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
        return model.predict(xgb.DMatrix(validation_exp),
                             ntree_limit=model.best_ntree_limit)

    def _get_importance(self,
                        model: Booster) -> DataFrame:
        """重要度を取得します

        Args:
            model (Booster): モデル

        Returns:
            DataFrame: _description_
        """
        score_dict = model.get_score(importance_type='gain')
        importance = DataFrame({
            'feature': list(score_dict.keys()),
            'gain': list(score_dict.values()),
            })

        return importance
