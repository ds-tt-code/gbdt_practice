from enum import Enum, auto


class Objective(Enum):
    """目的関数の列挙値です"""
    binary = auto()


class EvalMetric(Enum):
    """評価関数の列挙値です"""
    logloss = auto()
