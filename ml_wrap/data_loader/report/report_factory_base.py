"""レポート作成用ファクトリークラスです"""
from abc import ABCMeta, abstractmethod


class ReportFactoryBase(object, metaclass=ABCMeta):

    @staticmethod
    def get_factory(cls: type):
        factory = None

        for c in ReportFactoryBase.__subclasses__():
            if c is cls:
                factory = cls()

        if not factory:
            raise ValueError('指定されたクラスはReportFactoryBaseではありません')

        return factory

    @abstractmethod
    def create_report(self, title):
        pass

    @abstractmethod
    def create_paragraph(self, title):
        pass

    @abstractmethod
    def create_dataframe(self, title, df):
        pass

    @abstractmethod
    def create_target_basic_info(self, title, target):
        pass
