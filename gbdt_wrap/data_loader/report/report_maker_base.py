"""レポート作成クラスのベースクラスです"""
from abc import ABCMeta, abstractmethod

from gbdt_wrap.data_loader.target_data import TargetData


class ReportMakerBase(object, metaclass=ABCMeta):

    @staticmethod
    def get_report_maker(cls_str: str):
        for c in ReportMakerBase.__subclasses__():
            if c.__name__ == cls_str:
                return c

        raise ValueError('指定されたクラスはReportMakerBaseではありません')

    @staticmethod
    @abstractmethod
    def show_raw_data(data: TargetData):
        pass

    @staticmethod
    @abstractmethod
    def show_basic_info(data: TargetData):
        pass
