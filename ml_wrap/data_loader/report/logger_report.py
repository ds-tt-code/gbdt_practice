"""pythonのロギングを使ってレポートするクラス"""
from ml_wrap.data_loader.report.report_base import ReportBase


class LoggerReport(ReportBase):

    def __init__(self, title, logger):
        self.logger = logger
        super().__init__(title)

    def get_report(self):
        ret = f'======={self.title}========='
        ret += super().get_report()
        return ret

    def output(self):
        self.logger.info(self.get_report())
