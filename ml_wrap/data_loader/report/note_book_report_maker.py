"""JupyterNoteBook上でレポートを表示するクラス"""
import math
from numpy import float64
from IPython.display import display
from matplotlib.pyplot import show, figure, subplots_adjust, colorbar

from ml_wrap.data_loader.report.report_maker_base import ReportMakerBase
from ml_wrap.data_loader.target_data import TargetData


class NoteBookReportMaker(ReportMakerBase):

    @staticmethod
    def show_raw_data(data: TargetData):
        display(data._raw_data)

    @staticmethod
    def show_basic_info(data: TargetData):
        display(data.get_basic_info())

    @staticmethod
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

        tmp_cols = cols

        if len(tmp_cols) == 0:
            tmp_cols = list(data._raw_data.columns)

        fig = figure(figsize=figure_size)
        row = int(math.ceil(len(tmp_cols) / width))

        for idx, col_name in enumerate(tmp_cols):
            pos = row * 100 + width * 10 + idx + 1
            axis = fig.add_subplot(pos)
            if show_null:
                axis.hist(data._raw_data[col_name].astype(str), bins=bins)
            else:
                axis.hist(data._raw_data[col_name].dropna(), bins=bins)

            axis.set_title(col_name)
            for tick in axis.get_xticklabels():
                tick.set_rotation(x_label_rotate)

        subplots_adjust(hspace=margin_bottom, wspace=margin_left)
        show()

    @staticmethod
    def show_scatter(data: TargetData,
                     *cols, width=2,
                     figure_size=(10, 10),
                     margin_left=1,
                     margin_bottom=1,
                     x_label_rotate=90):

        if not data.target:
            raise Exception('目的変数の設定がありません')

        tmp_cols = cols

        if len(tmp_cols) == 0:
            tmp_cols = list(data._raw_data.columns)

        fig = figure(figsize=figure_size)
        row = int(math.ceil(len(tmp_cols) / width))

        for idx, col_name in enumerate(tmp_cols):
            pos = row * 100 + width * 10 + idx + 1
            axis = fig.add_subplot(pos)
            axis.scatter(data._raw_data[col_name],
                         data._raw_data[data.target],
                         alpha=0.1)
            axis.set_title(col_name)
            axis.set_xlabel(col_name)
            axis.set_ylabel(data.target)
            for tick in axis.get_xticklabels():
                tick.set_rotation(x_label_rotate)

        subplots_adjust(hspace=margin_bottom, wspace=margin_left)
        show()

    def show_correlation(data: TargetData):
        """相関係数を表示します"""
        return data._raw_data.corr()

    def show_correlation_heat(data: TargetData):

        fig = figure()

        axis = fig.add_subplot(111)
        cols = [col
                for col
                in data._raw_data.columns
                if data._raw_data[col].dtype == float64]

        heatmap = axis.pcolor(data._raw_data[cols].corr())
        axis.set_title('相関')
        axis.set_xticklabels(cols)
        axis.set_yticklabels(cols)
        for tick in axis.get_xticklabels():
            tick.set_rotation(90)

        colorbar(heatmap)
        show()
