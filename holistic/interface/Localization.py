from abc import ABC, abstractmethod

class Localization:
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def localize(self, questions):
        pass
