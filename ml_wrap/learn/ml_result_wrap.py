"""学習結果のベースクラス"""


class MlResultWrap(object):

    def __init__(self,
                 validation_exp,
                 validation_obj,
                 model,
                 label):
        """コンストラクタ"""
        self.validation_exp = validation_exp
        self.validation_obj = validation_obj
        self.model = model
        self.label = label
