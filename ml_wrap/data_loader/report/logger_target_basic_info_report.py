from ml_wrap.data_loader.report.logger_dataframe_report import LoggerDataFrameReport
from ml_wrap.data_loader.target_data import TargetData


class LoggerTargetBasicInfoReport(LoggerDataFrameReport):

    def __init__(self, title, target: TargetData):
        super().__init__(title, target.get_basic_info())
