"""JupyterNoteBook上でレポートを表示するクラス"""
from IPython.display import display

from gbdt_wrap.data_loader.report.report_maker_base import ReportMakerBase
from gbdt_wrap.data_loader.target_data import TargetData


class NoteBookReportMaker(ReportMakerBase):

    @staticmethod
    def show_raw_data(data: TargetData):
        display(data._raw_data)

    @staticmethod
    def show_basic_info(data: TargetData):
        display(data.get_basic_info())
