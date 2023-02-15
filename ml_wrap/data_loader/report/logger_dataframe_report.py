from ml_wrap.data_loader.report.report_dataframe_base import ReportDataFrameBase


class LoggerDataFrameReport(ReportDataFrameBase):

    def __init__(self, title, df):
        super().__init__(title, df)

    def get_report(self):
        return str(self.df)
