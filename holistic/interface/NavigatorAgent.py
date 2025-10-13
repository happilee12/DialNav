from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class NavigatorAgent(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def set_target_env(self, env_name):
        pass
    
    @abstractmethod
    def reset_epoch(self):
        pass
    
    @abstractmethod
    def set_next_batch(self):
        pass
    
    @abstractmethod
    def initialize_nav(self):
        pass

    @abstractmethod
    def get_next_action(self, nav_idx, obs):
        pass
    
    @abstractmethod
    def navigate(self, next_vp_ids, obs, traj):
        pass
    
    @abstractmethod
    def update_instruction(self, to_ask_indices, questions, answers):
        pass
    
    @abstractmethod
    def wta(self):
        pass
    
    @abstractmethod
    def ask(self):
        pass