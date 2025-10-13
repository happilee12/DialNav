from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class Navigation(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def navigate(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_instruction(self, to_ask_indices, questions, answers):
        pass

