"""和暦の列を西暦に変換します"""
import unicodedata
import re
from pandas import DataFrame

from gbdt_wrap.data_loader.processor.data_processor_base import DataProcessorBase


class JapaneseCalenderProcessor(DataProcessorBase):
    """すべてnullの列を削除します"""

    ERA_DICT = {
        "明治": 1868,
        "大正": 1912,
        "昭和": 1926,
        "平成": 1989,
        "令和": 2019,
    }

    def __init__(self, target, after_type=int):
        self.target = target
        self.after_type = after_type

    def _process(self, data: DataFrame) -> DataFrame:
        ret = data.copy()
        ret[self.target] = (data[self.target]
                            .map(self.convert)
                            .astype(self.after_type))
        return ret

    def convert(self, text: str) -> int:
        """指定された和暦を西暦に変換します

        Args:
            text (str): 変換対象の文字列

        Raises:
            ValueError: 和暦の形式でないものを指定された場合にエラーを発出します

        Returns:
            int: 西暦
        """

        # 正規化
        if str(text) != 'nan':
            text = unicodedata.normalize('NFKC', text)
            text = text.replace('元年', '1年')

            # 年月日を抽出
            era_list = self.ERA_DICT.keys()
            pattern = fr'(?P<era>{"|".join(era_list)})\s*(?P<year>[0-9]{{1,2}})\s*年?'
            m = re.search(pattern, text)

            # 抽出できなかったら終わり
            if m is None:
                return text
                # raise ValueError(f'指定された値を西暦に変換できませんでした。text={text}')

            # 年を変換
            return self.ERA_DICT[m.group('era')] + int(m.group('year')) - 1
