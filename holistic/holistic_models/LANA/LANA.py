import sys
import os
from argparse import Namespace
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(current_dir, '../../../modules/qa/LANA/finetune_src')
sys.path.insert(0, modules_path)

from interface.AnswerGeneration import AnswerGeneration
from interface.QuestionGeneration import QuestionGeneration
from .qg_default_args import get_qg_default_args
from .ag_default_args import get_ag_default_args
from transformers import BertTokenizer


try:
    from r2r.lana_speaker import LanaSpeaker
    from r2r.env import SpeakerBatch
    from r2r.data_utils import ImageFeaturesDB
    print("Successfully imported LANA")
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

class LANA(QuestionGeneration, AnswerGeneration):
    def __init__(self, basepath, args=None, rank=0, type='qg'):
        if type == 'qg':
            default_args = get_qg_default_args(basepath)
        elif type == 'ag':
            default_args = get_ag_default_args(basepath)
        args = merge_args(default_args, args)
        super().__init__(args)
        feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, use_clip16=args.use_clip16)
        self.language_env = SpeakerBatch(feat_db, args.connectivity_dir, args.scan_list, args.batch_size, args.angle_feat_size, args.seed)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.agent = LanaSpeaker(args, self.language_env, self.tokenizer)
        self.agent.vln_bert.eval()
        self.agent.critic.eval()
        if args.resume_file is not None:    
            print("########### [MODEL LOAD] Loading LANA model from", args.resume_file)
            self.agent.load(args.resume_file)

    def ask(self, scanIds, viewpoints):
        paths = [[viewpoint] for viewpoint in viewpoints]
        max_given_length = 0
        max_generate_length = 199
        created_tokens_list, natural_language_output, seen_paths = self.agent.say(scanIds, paths, env=self.language_env, max_given_length=max_given_length, max_generate_length=max_generate_length, given=[])
        
        for i in range(len(natural_language_output)):
            if len(natural_language_output[i]) == 0:
                natural_language_output[i] = 'Where should I go?'
        return natural_language_output, seen_paths
    
    
    def answer(self, scanIds, viewpoints, paths):
        max_given_length = 0
        max_generate_length = 199
        max_action_length = 20
        paths = [path[:max_action_length] for path in paths]
        created_tokens_list, natural_language_output, seen_paths = self.agent.say(scanIds, paths, env=self.language_env, max_given_length=max_given_length, max_generate_length=max_generate_length, given=[])

        for i, path in enumerate(paths):
            assert path == seen_paths[i], f"path: {path}, seen_paths: {seen_paths[i]}"
        return natural_language_output, seen_paths
