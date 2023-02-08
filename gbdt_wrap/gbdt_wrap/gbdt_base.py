"""GBDTのベースクラス"""
from gbdt_wrap.data_loader.data_loader_base import DataLoaderBase
from gbdt_wrap.gbdt_wrap.gbdt_meta import GBDTMetaClass


class GBDTBase(object, metaclass=GBDTMetaClass):
    """GBDTのベースクラスです"""
    NROUND = 100000
    ESR = 20
    LOGLEVEL = 5

    def __init__(self,
                 loader: DataLoaderBase,
                 seed: int = None):
        """コンストラクタ

        Args:
            loader (DataLoaderBase): _description_
            seed (int, optional): _description_. Defaults to None.
        """

        self.loader = loader
        self.seed = seed
