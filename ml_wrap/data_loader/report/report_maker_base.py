"""レポート作成クラスのベースクラスです"""
from abc import ABCMeta, abstractmethod

from ml_wrap.data_loader.target_data import TargetData


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

    @staticmethod
    @abstractmethod
    def show_histogram(data: TargetData,
                       *cols,
                       width=2,
                       bins=20,
                       show_null=False,
                       x_label_rotate=90,
                       figure_size=(10, 10),
                       font='MS Gothic',
                       margin_left=1,
                       margin_bottom=1):
        pass

    @staticmethod
    def define_colmns_enum(data: TargetData, *remove_char):
        """入力しやすいように列挙型を定義します。Enum名はDataColumns"""
        code = '''from enum import Enum
class DataColumns(Enum):
'''

        for col in data._raw_data.columns:
            enum_name = col
            for c in remove_char:
                enum_name = enum_name.replace(c, '')
            code += f' {enum_name}="{col}"\n'

        exec(code)

    @staticmethod
    @abstractmethod
    def show_scatter(data: TargetData,
                     *cols, width=2,
                     figure_size=(10, 10),
                     margin_left=1,
                     margin_bottom=1,
                     x_label_rotate=90):
        pass

    @staticmethod
    @abstractmethod
    def show_correlation(data: TargetData):
        """相関係数を表示します"""
        pass

    @staticmethod
    @abstractmethod
    def show_correlation_heat(data: TargetData):
        pass
