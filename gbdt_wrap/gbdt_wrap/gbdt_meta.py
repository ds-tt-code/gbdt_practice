"""GBDTのメタクラス"""


class GBDTMetaClass(type):
    """GBDTのメタクラスです"""

    def __new__(meta, name, bases, attributes):
        return type.__new__(meta, name, bases, attributes)
