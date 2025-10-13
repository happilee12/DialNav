from abc import ABC, abstractmethod

class AnswerGeneration(ABC):
    def __init__(self, args):
        self.args = args

    def answer(self):
        pass