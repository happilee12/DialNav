from abc import ABC, abstractmethod

class QuestionGeneration(ABC):
    def __init__(self, args):
        self.args = args

    def ask(self):
        pass