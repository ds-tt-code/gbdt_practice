"""ユーティリティ関数を定義するファイルです"""
import re
from pandas import Series
from logging import getLogger, NullHandler


def is_float(s):
    """指定された引数が数値変換可能かどうか判定します"""
    p = '[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?'
    return True if re.fullmatch(p, str(s)) else False


def get_not_float(col: Series) -> list[str]:
    """数値変換不可能な値を抽出します

    Args:
        col (Series): 抽出対象の列

    Returns:
        list[str]: 変換不可能な値
    """
    return [val for val in col.unique() if not is_float(str(val))]


def get_default_logger(name):
    """デフォルトのロガーを"""
    logger = getLogger(name)
    logger.addHandler(NullHandler)
