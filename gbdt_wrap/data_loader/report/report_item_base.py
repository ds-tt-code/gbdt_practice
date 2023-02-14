from abc import ABCMeta, abstractmethod


class ReportItemBase(object, metaclass=ABCMeta):
    """レポートの段落です"""

    def __init__(self, title):
        self.title = title
        self.contains = []

    @abstractmethod
    def get_report(self):
        pass

    def add_contains(self, contains):
        self.contains.append(contains)
