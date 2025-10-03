import sys
import os
from argparse import Namespace
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(current_dir, '../../../modules/loc/GCN')
sys.path.insert(0, modules_path)

from interface.Localization import Localization
from .default_args import get_default_args

try:
    from src.cross_modal.run import LEDAgent
    print("Successfully imported LEDAgent")
except ImportError as e:
    print(f"Import error: {e}")

def merge_args(default_args, new_args):
    """Merge new args with default args, only updating provided values"""
    if new_args is None:
        return default_args
    
    # Convert to dict for easier manipulation
    default_dict = vars(default_args)
    new_dict = vars(new_args) if hasattr(new_args, '__dict__') else new_args
    
    # Create merged dict
    merged_dict = default_dict.copy()
    
    # Only update values that are provided in new_args
    for key, value in new_dict.items():
        if value is not None:  # Only update if value is not None
            merged_dict[key] = value
    
    return Namespace(**merged_dict)

class GCNLocModel(Localization):
    def __init__(self, basepath, args=None, rank=0):
        default_args = get_default_args(basepath)
        args = merge_args(default_args, args)
        
        super().__init__(args)
        self.rank = rank
        self.agent = LEDAgent(args)
        self.agent.init_holistic_inference()
        self.agent.model.model.eval()
    
    def localize(self, scanIds, questions):
        return self.agent.localize(questions, scanIds)
    