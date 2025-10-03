import sys
import os
from argparse import Namespace
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(current_dir, '../../../modules/nav/ScaleVLN/map_nav_src')
sys.path.insert(0, modules_path)

from interface.Navigation import Navigation
from interface.WTA import WTA

import numpy as np
import torch.nn.functional as F


try:
    from nav_agent.agent import GMapNavAgent, GMapNavAgnetWta
    from sv_utils.data import ImageFeaturesDB
    from nav_agent.env import NDHNavBatch
    from .default_args import get_default_args
    print("Successfully imported nav_agent")
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

class ScaleVLNModel(Navigation, WTA):
    def __init__(self, basepath, args=None, rank=0):
        from transformers import AutoTokenizer
        default_args = get_default_args(basepath)
        args = merge_args(default_args, args)
        
        super().__init__(args)
        self.rank = rank
        self.agent = GMapNavAgnetWta(args, None, rank)
        # self.agent = GMapNavAgent(args, None, rank)
        self.feat_db = ImageFeaturesDB(self.args.val_ft_file, self.args.image_feat_size)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if args.resume_file is not None:
            self.args.resume_iter  = self.agent.load(args.resume_file)

        self.wta_question_threshold = args.nav_wta_question_threshold

        ### navigation status
        self.obs = None
        self.gmaps = None
        self.instruction = []
        self.instruction_encoded = []
        self.language_inputs = None
        self.txt_embeds = None
        self.ended = None
    
    
    def eval(self):
        self.agent.vln_bert.eval()
        self.agent.critic.eval()

    def set_envs(self, envs, instr_data_dict):
        self.val_envs = {}
        for env in envs:
            if env not in instr_data_dict:
                raise ValueError(f"Environment {env} not found in instr_data_dict")
            self.val_envs[env] = NDHNavBatch(self.feat_db,
                          instr_data_dict[env], 
                          self.args.connectivity_dir,
                          batch_size=self.args.batch_size, 
                          angle_feat_size=self.args.angle_feat_size, 
                          seed=self.args.seed, name=env)
    
    def set_target_env(self, env_name):
        if env_name not in self.val_envs:
            raise ValueError(f"Environment {env_name} not found in val_envs")
        self.agent.env = self.val_envs[env_name]

    def reset_epoch(self):
        self.agent.env.reset_epoch(shuffle=False)
    
    def _get_state(self):
        return self.agent.env.env.getStates()
    
    def set_next_batch(self):
        kwargs = {'holistic': True}
        self.agent.env.reset(**kwargs)

    def get_obs(self):
        obs = []
        navigation_env = self.agent.env
        for i, (feature, state) in enumerate(self._get_state()):
            item = navigation_env.batch[i]
            base_view_id = state.viewIndex
           
            # Full features
            candidate = navigation_env.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, navigation_env.angle_feature[base_view_id]), -1)

            ob = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instruction' : item['instruction'],
                'instr_encoding': item['instr_encoding'],
                'gt_path' : item['path'],
                'path_id' : item['path_id'],
                'end_panos' : item['end_panos'],
            }
            # # RL reward. The negative distance between the state and the final state
            # # There are multiple gt end viewpoints on REVERIE. 
            # if ob['instr_id'] in self.gt_trajs:
            #     ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            # else:
            #     ob['distance'] = 0

            obs.append(ob)
        return obs

    def initialize_nav(self, obs):
        self.agent.feedback = 'argmax'
        self.gmaps = self.agent._initialize_graph(obs)
        self.ended = np.array([False] * len(obs))
        self.agent._update_graph_structure(obs, self.gmaps, self.ended)
        self.instruction = [ob['instruction'] for ob in obs]
        self.instruction_encoded = [self.tokenizer.encode(instr) for instr in self.instruction]
        self.language_inputs, self.txt_embeds = self.agent._set_instruction(self.instruction_encoded)
        self.agent._update_scanvp_cands(obs)

    def get_next_action(self, nav_idx, obs):
        nav_inputs, pano_inputs = self.agent._process_navigation_step(obs, self.gmaps, self.ended, nav_idx, self.language_inputs, self.txt_embeds)
        nav_probs, nav_vpids, nav_logits, nav_outs = self.agent._nav_probs(nav_inputs)
        self.agent._update_stop_scores(nav_probs, self.gmaps, obs, self.ended)

        # decide next action. feedback = argmax
        _, _, next_vp_ids, ended, _ = self.agent.decide_action(nav_logits, nav_probs, nav_vpids, nav_inputs, obs, self.ended, len(obs), nav_idx)
        instrucion_for_this_nav = [self.tokenizer.decode(seq[seq.nonzero().squeeze()]) for seq in self.language_inputs['txt_ids']]
        return next_vp_ids, ended, nav_probs, instrucion_for_this_nav, nav_outs
    
    def navigate(self, next_vp_ids, obs, just_ended, traj):
        ## TODO refactor - do not pass traj

        previous_vp_ids = [ob['viewpoint'] for ob in obs]
        self.agent.make_equiv_action(next_vp_ids, self.gmaps, obs, traj)
        new_obs = self.get_obs()
        self.agent._update_stop_node(new_obs, self.gmaps, self.ended, just_ended, traj, len(obs))
        self.agent._update_graph_structure(new_obs, self.gmaps, self.ended)
        # self.ended[:] = np.logical_or(self.ended, np.array([x is None for x in next_vp_ids]))
        self.ended[:] = np.logical_or(self.ended, just_ended)

        paths = [self.gmaps[i].graph.path(previous_vp_ids[i], next_vp_ids[i]) for i in range(len(obs))]
        return new_obs, paths

    def update_instruction(self, to_ask_indices, questions, answers):
        obs = self.get_obs()
        new_instructions = []
        new_instructions_encoded = []
        for i, ob in enumerate(obs):
            new_instructions.append(f"{answers[i]} {ob['instruction']}")
            target_encoded = self.tokenizer.encode(ob["instruction"])
            answer_encoded = self.tokenizer.encode(answers[i])
            new_instructions_encoded.append(answer_encoded + target_encoded[1:])

        for i in to_ask_indices:
            self.instruction[i] = new_instructions[i]
            self.instruction_encoded[i] = new_instructions_encoded[i]

        # print("updated instruction", self.instruction, "updated indices", to_ask_indices)
        self.language_inputs, self.txt_embeds = self.agent._set_instruction(self.instruction_encoded)

    def wta(self, step, nav_probs, nav_outs):
        wta_logits = nav_outs['question_logits']
        wta_probs = F.softmax(wta_logits, dim=1)

        # _, ask = wta_probs.max(1)
        # ask = ask.cpu().numpy()  # Move the tensor to CPU and convert to NumPy array
        # ask = np.logical_and(np.logical_not(self.ended), ask) 

        prob_ask = wta_probs[:, 1]  
        ask = prob_ask > self.wta_question_threshold
        ask = ask.cpu().numpy()  # Move the tensor to CPU and convert to NumPy array
        ask = np.logical_and(np.logical_not(self.ended), ask) 
 
        return ask
    