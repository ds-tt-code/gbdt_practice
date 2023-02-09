"""GBDTのベースクラス"""
from abc import abstractmethod

import numpy as np
from pandas import DataFrame, concat
from sklearn.metrics import log_loss

from gbdt_wrap.data_loader.data_loader_base import DataLoaderBase
from gbdt_wrap.data_loader.data_processor_base import DataProcessorBase
from gbdt_wrap.gbdt_wrap.gbdt_meta import GBDTMetaClass


class GBDTBase(object, metaclass=GBDTMetaClass):
    """GBDTのベースクラスです"""
    NROUND = 100000  # 基本的は大きい値で固定。これで調整することはない
    ESR = 20
    LOGLEVEL = 5

    def __init__(self,
                 loader: DataLoaderBase,
                 data_processor: list[DataProcessorBase],
                 seed: int = None):
        """コンストラクタ

        Args:
            loader (DataLoaderBase): _description_
            seed (int, optional): _description_. Defaults to None.
        """

        self.data = loader.load()
        for p in data_processor:
            p.process(self.data)
        self.seed = seed

    def learn_cv(self, n_fold):
        """k-fold Cross Validationで学習を実行します

        Args:
            n_fold (_type_): k-fold cross validation のfold数

        Returns:
            _type_: importance, loglossの平均, preds(予測)
        """
        importances = DataFrame()
        scores_logloss = []
        oof_preds = np.zeros(len(self.data.obj_val))

        for fold, (train_index, validation_index) in enumerate(
                                                self.data.get_nfold_indexes(
                                                    n_fold, self.seed
                                                )
                                              ):
            train_exp, train_obj = \
                self.data.get_train_data(train_index)

            validation_exp, validation_obj = \
                self.data.get_validation_data(validation_index)

            importance, logloss, val_preds = \
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
            _type_: importance, loglossの平均, preds(予測)
        """

        train_exp_ds, validation_exp_ds = \
            self._trans_data(train_exp,
                             train_obj,
                             validation_exp,
                             validation_obj)

        model = self._get_model(train_exp_ds, validation_exp_ds)
        val_preds = self._predict(model, validation_exp)
        importance = self._get_importance(model)
        logloss = log_loss(validation_obj, val_preds)
        return importance, logloss, val_preds

    @abstractmethod
    def _trans_data(self,
                    train_exp,
                    train_obj,
                    validation_exp,
                    validation_obj):
        """ データセットの定義
            XGBoostはこの処理を挟む事で省メモリに学習する事ができる

        Args:
            train_exp (_type_): _description_
            train_obj (_type_): _description_
            validation_exp (_type_): _description_
            validation_obj (_type_): _description_

        Returns:
            _type_: _description_
        """

    @abstractmethod
    def _get_model(self,
                   train_data,
                   validation_data):
        """モデルを生成します

        Args:
            train_data (xgb.DMatrix): 訓練用データ
            validation_data (xgb.DMatrix): 検証用データ

        Returns:
            _type_: _description_
        """
        pass

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

    @abstractmethod
    def _get_importance(self,
                        model):
        """重要度を取得します

        Args:
            model (_type_): _description_
        """
        pass
