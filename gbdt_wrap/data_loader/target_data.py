"""学習対象のデータを表すクラス"""
from logging import getLogger

from pandas import DataFrame, Series


class TargetData(object):
    """学習対象のデータを表すクラス"""

    def __init__(self,
                 data: DataFrame,
                 target: str = None,
                 categories: list[str] = []):
        """コンストラクタです

        Args:
            data (DataFrame): 対象のデータ
            target(str): 目的変数のカラム名
            categories (list[str]): カテゴリ変数のリスト
        """
        self._raw_data = data
        self.target = target
        self.categories = categories
        self.logger = getLogger(__name__)

    def get_basic_info(self, unique_max=10) -> DataFrame:
        """データの基本情報を取得します"""
        # 行数と列数を取得
        row_count = len(self._raw_data)

        result = DataFrame(index=self._raw_data.columns)
        non_null_count = self._raw_data.count()
        result['types'] = self._raw_data.dtypes
        result['count'] = row_count
        result['non-null-count'] = non_null_count
        result['null-count'] = result['count'] - result['non-null-count']
        result['null%'] = result['null-count'] / result['count'] * 100
        result['max'] = self._raw_data.max(numeric_only=True)
        result['min'] = self._raw_data.min(numeric_only=True)
        result['mean'] = self._raw_data.mean(numeric_only=True)
        result['std'] = self._raw_data.std(numeric_only=True)
        result['unique-count'] = self._raw_data.nunique()
        result['top10 val'] = Series({col: ','.join([str((idx,
                                                         val,
                                                         f'{round(val/row_count * 100)}%'))
                                                    for idx, val
                                                    in self._raw_data[col].value_counts().items()][:unique_max])
                                      for col in self._raw_data.columns})

        return result
