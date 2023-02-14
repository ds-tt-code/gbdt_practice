"""データをレポートするクラス"""
from abc import ABCMeta, abstractmethod

from gbdt_wrap.data_loader.report.report_item_base import ReportItemBase


class ReportBase(ReportItemBase, metaclass=ABCMeta):
    """データをレポートするクラスです"""

    def __init__(self, title):
        self.title = title
        super().__init__(title)

    def get_report(self):
        """レポートの内容を取得します"""
        return '\n'.join([r.get_report() for r in self.contains])

    @abstractmethod
    def output(self):
        pass
