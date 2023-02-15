from ml_wrap.data_loader.report.report_item_base import ReportItemBase


class ReportTerminateItemBase(ReportItemBase):

    def __init__(self, title):
        super().__init__(title)

    def add_contains(self, contains):
        raise Exception('末端要素のため内容を追加できません')
