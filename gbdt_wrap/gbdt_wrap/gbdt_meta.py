"""GBDTのメタクラス"""
from abc import ABCMeta


class GBDTMetaClass(ABCMeta):
    """GBDTのメタクラスです"""

    def __new__(meta, name, bases, attributes):
        return type.__new__(meta, name, bases, attributes)
