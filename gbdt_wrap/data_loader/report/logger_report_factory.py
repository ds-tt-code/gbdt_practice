from logging import getLogger, StreamHandler, INFO

from gbdt_wrap.data_loader.report.report_factory_base import ReportFactoryBase
from gbdt_wrap.data_loader.report.logger_report import LoggerReport
from gbdt_wrap.data_loader.report.logger_target_basic_info_report import LoggerTargetBasicInfoReport


class LoggerReportFactory(ReportFactoryBase):

    def __init__(self, logger=None):
        self.logger = logger or getLogger()
        self.logger.handlers.append(StreamHandler())
        self.logger.setLevel(INFO)

    def create_report(self, title):
        return LoggerReport(title, self.logger)

    def create_paragraph(self):
        pass

    def create_dataframe(self, title, df):
        pass

    def create_target_basic_info(self, title, target):
        return LoggerTargetBasicInfoReport(title, target)
