from gbdt_wrap.data_loader.report.report_item_base import ReportItemBase


class ReportParagraphBase(ReportItemBase):

    def __init__(self, title):
        super().__init__(title)
        self.contains = []
