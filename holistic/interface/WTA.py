from abc import ABC, abstractmethod

class WTA(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def wta(self, *args, **kwargs):
        pass