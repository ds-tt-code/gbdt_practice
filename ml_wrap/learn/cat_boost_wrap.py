"""CatBoostのラップクラスです"""
import catboost as cb
from pandas import DataFrame

from ml_wrap.data_loader.processor.category_na_padding_processor import CategoryProcessorBase
from ml_wrap.learn.data_def import EvalMetric, Objective
from ml_wrap.learn.gbdt_base import GBDTBase
from ml_wrap.data_loader.loader import TargetData
from ml_wrap.learn.divider.data_divider_base import DataDividerBase


class CatBoostWrap(GBDTBase):

    _OBJECTIVE_PARAM = {
        Objective.binary: 'Logloss'
    }

    _EVAL_METRIC_PARAM = {
        EvalMetric.logloss: 'Logloss'
    }

    def __init__(self,
                 result_label: str = None,
                 objective: Objective = Objective.binary,
                 eval_metric: EvalMetric = EvalMetric.logloss,
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
        super().__init__(CategoryProcessorBase('__NA__'),
                         result_label,
                         early_stopping_rounds=early_stopping_rounds)

        self.params = {
            'objective': self._OBJECTIVE_PARAM[objective],
            'eval_metric': self._EVAL_METRIC_PARAM[eval_metric],
            'learning_rate': 0.01,
            'max_depth': max_depth,
        }

        if seed:
            self.params['random_state'] = seed

    def learn(self, data: TargetData, divider: DataDividerBase):
        # CatBoostでカテゴリ変数の機能を使うための処理
        # カラムのインデクスを渡す
        self.cat_col_idx = [self.data.exp_val.columns.get_loc(cat)
                            for cat in self.data.categories]
        super().learn(data, divider)

    def _learn(self,
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

        train_exp_ds, validation_exp_ds = \
            cb.Pool(train_exp,
                    label=train_obj,
                    cat_features=self.cat_col_idx), \
            cb.Pool(validation_exp,
                    label=validation_obj,
                    cat_features=self.cat_col_idx)

        return cb.train(
            params=self.params,
            dtrain=train_exp_ds,
            num_boost_round=self.N_ROUND,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_set=validation_exp_ds,
            verbose_eval=self.LOGLEVEL
        )

    def _predict(self, model, validation_exp: DataFrame):
        """Out of Fold法を用いて訓練データの予測を行う

        Args:
            model (any): catboostのモデル
            validation_exp (DataFrame): _description_

        Returns:
            _type_: _description_
        """
        # ２次元配列で返ってくるので1になる確率を取得する
        ret = model.predict(validation_exp,
                            prediction_type='Probability',
                            ntree_end=model.best_iteration_)
        return ret[:, 1]

    def calc_importance(self, model):
        """重要度を取得します

        Args:
            model (any): catboostモデル

        Returns:
            DataFrame: _description_
        """
        return {
            'feature': list(model.feature_names_),
            'gain': list(model.feature_importances_),
        }
