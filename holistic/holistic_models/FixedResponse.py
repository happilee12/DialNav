from interface.AnswerGeneration import AnswerGeneration
from interface.QuestionGeneration import QuestionGeneration


class FixedAnswerGeneration(AnswerGeneration):
    def __init__(self, response=''):
        self.response = response

    def answer(self, to_ask_indices, questions, viewpoints):
        return [self.response for _ in range(len(viewpoints))]

class FixedQuestionGeneration(QuestionGeneration):
    def __init__(self, question=''):
        self.question = question

    def ask(self):
        return self.question