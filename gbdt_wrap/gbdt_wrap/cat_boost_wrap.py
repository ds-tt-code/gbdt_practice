"""CatBoostのラップクラスです"""
import catboost as cb
from pandas import DataFrame
from xgboost import Booster

from gbdt_wrap.data_loader.data_loader_base import DataLoaderBase
from gbdt_wrap.data_loader.na_padding_processor import NaPaddingProcessor
from gbdt_wrap.gbdt_wrap.data_def import EvalMetric, Objective
from gbdt_wrap.gbdt_wrap.gbdt_base import GBDTBase


class CatBoostWrap(GBDTBase):

    _OBJECTIVE_PARAM = {
        Objective.binary: 'Logloss'
    }

    _EVAL_METRIC_PARAM = {
        EvalMetric.logloss: 'Logloss'
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
        processor = [NaPaddingProcessor(loader.categories, '__NA__')]

        super().__init__(loader, processor, seed)

        self.params = {
            'objective': self._OBJECTIVE_PARAM[objective],
            'eval_metric': self._EVAL_METRIC_PARAM[eval_metric],
            'learning_rate': 0.01,
            'max_depth': max_depth,
        }

        if seed:
            self.params['random_state'] = seed

        # CatBoostでカテゴリ変数の機能を使うための処理
        # カラムのインデクスを渡す
        self.cat_col_idx = [self.data.exp_val.columns.get_loc(cat)
                            for cat in self.data.categories]

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

        return cb.Pool(train_exp,
                       label=train_obj,
                       cat_features=self.cat_col_idx), \
            cb.Pool(validation_exp,
                    label=validation_obj,
                    cat_features=self.cat_col_idx)

    def _get_model(self,
                   train_data: cb.Pool,
                   validation_data: cb.Pool) -> Booster:
        """モデルを生成します

        Args:
            train_data (cb.Pool): 訓練用データ
            validation_data (cb.Pool): 検証用データ

        Returns:
            _type_: _description_
        """

        return cb.train(
            params=self.params,
            dtrain=train_data,
            num_boost_round=self.NROUND,
            early_stopping_rounds=self.ESR,
            eval_set=validation_data,
            verbose_eval=self.LOGLEVEL
        )

    def _predict(self,
                 model,
                 validation_exp: DataFrame):
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

    def _get_importance(self,
                        model) -> DataFrame:
        """重要度を取得します

        Args:
            model (any): catboostモデル

        Returns:
            DataFrame: _description_
        """
        importance = DataFrame({
            'feature': list(model.feature_names_),
            'gain': list(model.feature_importances_),
            })

        return importance
