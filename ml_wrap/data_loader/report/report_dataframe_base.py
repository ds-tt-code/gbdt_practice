from pandas import DataFrame

from ml_wrap.data_loader.report.report_teminate_item_base import ReportTerminateItemBase


class ReportDataFrameBase(ReportTerminateItemBase):

    def __init__(self, title, df: DataFrame):
        self.title = title
        self.df = df
