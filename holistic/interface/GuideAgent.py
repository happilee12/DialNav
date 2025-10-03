from interface.AnswerGeneration import AnswerGeneration
from interface.Localization import Localization

class GuideAgent:
    def __init__(self, args, answer_model: AnswerGeneration, localization_model: Localization):
        self.args = args
        self.answer_model = answer_model
        self.localization_model = localization_model
    
    def answer(self):
        if self.answer_model is None:
            raise ValueError("answer_model is not set")
        return self.answer_model.generate()
    
    def localize(self):
        if self.localization_model is None:
            raise ValueError("localization_model is not set")
        return self.localization_model.localize()
    