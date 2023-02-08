"""XGBoostのラップクラスです"""
import numpy as np
import xgboost as xgb
from pandas import DataFrame, concat
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import Booster

from gbdt_wrap.gbdt_wrap.data_def import Objective, EvalMetric
from gbdt_wrap.data_loader.data_loader_base import DataLoaderBase
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

        # XGBoostの場合はカテゴリ変数に処理をする
        self.category_process()

    def learn_cv(self, n_fold):
        """k-fold Cross Validationで学習を実行します

        Args:
            n_fold (_type_): k-fold cross validation のfold数

        Returns:
            _type_: importance, loglossの平均, preds(予測)
        """
        importances = DataFrame()
        scores_logloss = []
        oof_preds = np.zeros(len(self.loader.obj_val))

        for fold, (train_index, validation_index) in enumerate(
                                                self.loader.get_nfold_indexes(
                                                    n_fold, self.seed
                                                )
                                              ):
            train_exp, train_obj = \
                self.loader.get_train_data(train_index)

            validation_exp, validation_obj = \
                self.loader.get_validation_data(validation_index)

            val_preds, importance, logloss = \
                self.learn(train_exp, train_obj, validation_exp, validation_obj)

            oof_preds[validation_index] = val_preds
            importance['fold'] = fold
            importances = concat([importances, importance], axis=0)
            scores_logloss.append(logloss)

        logloss = np.mean(scores_logloss)
        return importances, logloss, oof_preds

    def learn(self,
              train_exp,
              train_obj,
              validation_exp,
              validation_obj):
        """学習を実行します

        Args:
            train_exp (_type_): _description_
            train_obj (_type_): _description_
            validation_exp (_type_): _description_
            validation_obj (_type_): _description_

        Returns:
            _type_: _description_
        """

        train_exp, validation_exp = \
            self._trans_to_xgboost_data(train_exp,
                                        train_obj,
                                        validation_exp,
                                        validation_obj)

        model = self._get_model(train_exp, validation_exp)
        val_preds = self._predict(model, validation_exp)
        importance = self._get_impotance(model)
        logloss = log_loss(validation_obj, val_preds)
        return val_preds, importance, logloss

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

    def _trans_to_xgboost_data(self,
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
                             ntree_limit=model.best_ntree_limit)

    def _get_impotance(self,
                       model: Booster):
        score_dict = model.get_score(importance_type='gain')
        importance = DataFrame({
            'feature': list(score_dict.keys()),
            'gain': list(score_dict.values()),
            })

        return importance

    def category_process(self):
        """カテゴリ変数処理を実行します"""

        le = LabelEncoder()
        for cat in self.loader.categories:
            self.loader.exp_val[cat] = le.fit_transform(
                                    self.loader.exp_val[cat]
                                )
