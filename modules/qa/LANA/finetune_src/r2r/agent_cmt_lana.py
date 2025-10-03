import json
import os
import sys

import networkx as nx
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.misc import length2mask
from utils.logger import print_progress

from lana_models.model_lana import VLNBertCMT, Critic
from lana_models.tokenization_clip import SimpleTokenizer

from .eval_utils import cal_dtw

from .agent_base import BaseAgent
from r2r.env import SpeakerBatch
from tqdm import tqdm
from utils.logger import write_to_record_file, print_progress, timeSince

class Seq2SeqCMTAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, tok, rank=0):
        super().__init__(env)
        self.args = args

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # Models
        self._build_model()

        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device('cuda:%d' % self.rank)  # TODO

        # Optimizers
        if self.args.optim == 'rms':
            optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD
        else:
            assert False
        # if self.default_gpu:
        #     print('Optimizer: %s' % self.args.optim)

        param_optimizer = list(self.vln_bert.named_parameters())

        params = [{'params': [p for n, p in param_optimizer if 'clip' in n], "lr": self.args.clip_lr}, 
                  {'params': [p for n, p in param_optimizer if 'clip' not in n]}]

        self.vln_bert_optimizer = optimizer(params, lr=self.args.lr)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, size_average=False)

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
        
        self.tokenizer = tok

        if args.use_clip16:
            self.tokenizer = SimpleTokenizer(bpe_path=args.bpe_path)
            self.cls_token_id = 49406
            self.sep_token_id = 49407
            self.pad_token_id = 0
            self.mask_token_id = 40409          # HARD CODE
    
        self.action_cache = {}
        if self.args.use_cache:
            self.cache_keys = set()
            if self.args.cache_type == 'disk':
                self.load_cache_keys()
            print("Loaded cache keys", len(self.cache_keys))

    def _build_model(self):
        self.vln_bert = VLNBertCMT(self.args).cuda()
        self.critic = Critic(self.args).cuda()

    def _language_variable(self, obs):
        
        GIVEN_TEXT = False 
        if self.args.max_given_len > 0:
            GIVEN_TEXT = True
            GIVEN_LENGTH = self.args.max_given_len 
        

        if GIVEN_TEXT:
            seq_lengths = [len(ob['given_encoding']) + len(ob['instr_encoding']) for ob in obs]
            instr_lengths = [len(ob['instr_encoding']) for ob in obs]
            seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
            mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
            for i, ob in enumerate(obs):
                seq_tensor[i, :GIVEN_LENGTH] = ob['given_encoding']
                # seq_tensor[i, GIVEN_LENGTH] = self.sep_token_id
                seq_tensor[i, GIVEN_LENGTH:GIVEN_LENGTH+instr_lengths[i]] = ob['instr_encoding']
                mask[i, :GIVEN_LENGTH+instr_lengths[i]] = True

            seq_tensor = torch.from_numpy(seq_tensor)
            mask = torch.from_numpy(mask)
            return seq_tensor.long().cuda(), mask.cuda(), seq_lengths

        else: 
            seq_lengths = [len(ob['instr_encoding']) for ob in obs]
            seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
            mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
            for i, ob in enumerate(obs):
                seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
                mask[i, :seq_lengths[i]] = True

            seq_tensor = torch.from_numpy(seq_tensor)
            mask = torch.from_numpy(mask)
            return seq_tensor.long().cuda(), mask.cuda(), seq_lengths

    def _cand_pano_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        ob_cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        ob_lens = []
        ob_img_fts, ob_ang_fts, ob_nav_types, ob_pos = [], [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            cand_img_fts, cand_ang_fts, cand_nav_types, cand_pos = [], [], [], []
            cand_pointids = np.zeros((self.args.views,), dtype=np.bool)
            for j, cc in enumerate(ob['candidate']):
                cand_img_fts.append(cc['feature'][:self.args.image_feat_size])
                cand_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                cand_pointids[cc['pointId']] = True
                cand_nav_types.append(1)
                if self.args.cand_use_ob_pos:
                    cand_pos.append(ob['position'])
                else:
                    cand_pos.append(cc['position'])
            # add [STOP] feature
            cand_img_fts.append(np.zeros((self.args.image_feat_size,), dtype=np.float32))
            cand_ang_fts.append(np.zeros((self.args.angle_feat_size,), dtype=np.float32))
            cand_pos.append(ob['position'])
            cand_img_fts = np.vstack(cand_img_fts)
            cand_ang_fts = np.vstack(cand_ang_fts)
            cand_nav_types.append(2)

            # add pano context
            pano_fts = ob['feature'][~cand_pointids]
            cand_pano_img_fts = np.concatenate([cand_img_fts, pano_fts[:, :self.args.image_feat_size]], 0)
            cand_pano_ang_fts = np.concatenate([cand_ang_fts, pano_fts[:, self.args.image_feat_size:]], 0)
            cand_nav_types.extend([0] * (self.args.views - np.sum(cand_pointids)))
            cand_pos.extend([ob['position'] for _ in range(self.args.views - np.sum(cand_pointids))])

            ob_lens.append(len(cand_nav_types))
            ob_img_fts.append(cand_pano_img_fts)
            ob_ang_fts.append(cand_pano_ang_fts)
            ob_nav_types.append(cand_nav_types)
            ob_pos.append(cand_pos)

        # pad features to max_len
        max_len = max(ob_lens)
        for i in range(len(obs)):
            num_pads = max_len - ob_lens[i]
            ob_img_fts[i] = np.concatenate([ob_img_fts[i], \
                                            np.zeros((num_pads, ob_img_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_ang_fts[i] = np.concatenate([ob_ang_fts[i], \
                                            np.zeros((num_pads, ob_ang_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0] * num_pads)
            ob_pos[i] = np.array(ob_pos[i] + [np.array([0, 0, 0], dtype=np.float32) for _ in range(num_pads)])

        ob_img_fts = torch.from_numpy(np.stack(ob_img_fts, 0)).cuda()
        ob_ang_fts = torch.from_numpy(np.stack(ob_ang_fts, 0)).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()
        ob_pos = torch.from_numpy(np.stack(ob_pos, 0)).float().cuda()

        return ob_img_fts, ob_ang_fts, ob_nav_types, ob_lens, ob_cand_lens, ob_pos

    def _candidate_variable(self, obs):
        cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        max_len = max(cand_lens)
        cand_img_feats = np.zeros((len(obs), max_len, self.args.image_feat_size), dtype=np.float32)
        cand_ang_feats = np.zeros((len(obs), max_len, self.args.angle_feat_size), dtype=np.float32)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int64)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                cand_img_feats[i, j] = cc['feature'][:self.args.image_feat_size]
                cand_ang_feats[i, j] = cc['feature'][self.args.image_feat_size:]
                cand_nav_types[i, j] = 1
            cand_nav_types[i, cand_lens[i] - 1] = 2

        cand_img_feats = torch.from_numpy(cand_img_feats).cuda()
        cand_ang_feats = torch.from_numpy(cand_ang_feats).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()
        return cand_img_feats, cand_ang_feats, cand_nav_types, cand_lens

    def _history_variable(self, obs):
        hist_img_feats = np.zeros((len(obs), self.args.image_feat_size), np.float32)
        for i, ob in enumerate(obs):
            hist_img_feats[i] = ob['feature'][ob['viewIndex'], :self.args.image_feat_size]
        hist_img_feats = torch.from_numpy(hist_img_feats).cuda()

        if self.args.hist_enc_pano:
            hist_pano_img_feats = np.zeros((len(obs), self.args.views, self.args.image_feat_size), np.float32)
            hist_pano_ang_feats = np.zeros((len(obs), self.args.views, self.args.angle_feat_size), np.float32)
            for i, ob in enumerate(obs):
                hist_pano_img_feats[i] = ob['feature'][:, :self.args.image_feat_size]
                hist_pano_ang_feats[i] = ob['feature'][:, self.args.image_feat_size:]
            hist_pano_img_feats = torch.from_numpy(hist_pano_img_feats).cuda()
            hist_pano_ang_feats = torch.from_numpy(hist_pano_ang_feats).cuda()
        else:
            hist_pano_img_feats, hist_pano_ang_feats = None, None

        return hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats

    def _teacher_action(self, obs, ended, max_hist_len):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:  # Next view point
                        a[i] = k + max_hist_len
                        break
                else:  # Stop here
                    assert ob['teacher'] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate']) + max_hist_len
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, obs, max_hist_len, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """

        def take_action(i, name):
            if type(name) is int:  # Go to the next view
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:  # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        def adjust(i, src_point, trg_point):
            src_level = src_point // 12  # The point idx started from 0
            trg_level = trg_point // 12
            while src_level < trg_level:  # Tune up
                take_action(i, 'up')
                src_level += 1
            while src_level > trg_level:  # Tune down
                take_action(i, 'down')
                src_level -= 1
            while self.env.env.sims[i].getState()[0].viewIndex != trg_point:  # Turn right until the target
                take_action(i, 'right')

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:  # -1 is the <stop> action
                if action >= max_hist_len:
                    action = action - max_hist_len
                    select_candidate = ob['candidate'][action]
                    src_point = ob['viewIndex']
                    trg_point = select_candidate['pointId']
                    adjust(i, src_point, trg_point)
                    state = self.env.env.sims[i].getState()[0]
                    for idx, loc in enumerate(state.navigableLocations):
                        if loc.viewpointId == select_candidate['viewpointId']:
                            take_action(i, idx)
                            state = self.env.env.sims[i].getState()[0]
                            if traj is not None:
                                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
                else:
                    action = action - 1  # 1 for global history token
                    target_vp = self.seq_vp_list[i][action]
                    current_vp = ob['viewpoint']
                    src_point = ob['viewIndex']
                    trg_point = self.seq_view_idx_list[i][action]
                    path = nx.single_source_shortest_path(self.graphs[i], current_vp)[target_vp]
                    state = self.env.env.sims[i].getState()[0]
                    for j in range(len(path) - 1):
                        # from path[j] to path[j+1]
                        for idx, loc in enumerate(state.navigableLocations):
                            if loc.viewpointId == path[j+1]:
                                take_action(i, idx)
                                state = self.env.env.sims[i].getState()[0]
                                if traj is not None:
                                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
                                break
                        else:
                            raise ValueError('no navigable location')
                    adjust(i, src_point, trg_point)

    def _init_graph(self, batch_size):
        self.graphs = [nx.Graph() for _ in range(batch_size)]
        self.vp2idx_list = [dict() for _ in range(batch_size)]
        self.seq_idx_list = [list() for _ in range(batch_size)]
        self.seq_vp_list = [list() for _ in range(batch_size)]
        self.seq_view_idx_list = [list() for _ in range(batch_size)]
        self.seq_dist_list = [list() for _ in range(batch_size)]
        self.seq_dup_vp = [list() for _ in range(batch_size)]
        self.seq_last_idx = [dict() for _ in range(batch_size)]
        self.blocked_path = [defaultdict(lambda: defaultdict(lambda: 0)) for _ in range(batch_size)]

    def _update_graph(self, obs, ended, a_t, max_hist_len):
        for i, ob in enumerate(obs):
            if ended[i]:
                self.seq_dup_vp[i].append(True)
                continue
            vp = ob['viewpoint']
            if vp not in self.vp2idx_list[i]:
                idx = len(self.vp2idx_list[i])
                self.vp2idx_list[i][vp] = idx
                self.graphs[i].add_node(vp)
                self.seq_dup_vp[i].append(False)
                self.seq_last_idx[i][vp] = len(self.seq_dup_vp[i]) - 1
            else:
                idx = self.vp2idx_list[i][vp]
                if self.args.no_temporal_strategy == 'replace':
                    self.seq_dup_vp[i].append(False)
                    self.seq_dup_vp[i][self.seq_last_idx[i][vp]] = True
                else:  # 'keep' strategy, keep the old one
                    self.seq_dup_vp[i].append(True)
                self.seq_last_idx[i][vp] = len(self.seq_dup_vp[i]) - 1
            self.seq_idx_list[i].append(idx)
            self.seq_vp_list[i].append(vp)
            self.seq_view_idx_list[i].append(ob['viewIndex'])
            self.seq_dist_list[i].append(ob['distance'])
            for adj in ob['navigableLocations']:
                adj_vp = adj.viewpointId
                if adj_vp in self.vp2idx_list[i]:
                    self.graphs[i].add_edge(vp, adj_vp)

            # block path if backtrack
            if max_hist_len > a_t[i] >= 0:
                hist_vp = self.seq_vp_list[i][a_t[i] - 1]
                self.blocked_path[i][hist_vp][vp] += 1

    def _get_connectivity_mask(self):
        batch_size = len(self.graphs)
        max_size = max([len(seq) for seq in self.seq_idx_list])
        mask = torch.ones((batch_size, max_size, max_size)).cuda()
        for i in range(batch_size):
            adj_matrix = nx.adj_matrix(self.graphs[i], weight=1)
            adj_matrix.setdiag(1)
            adj_matrix = adj_matrix.toarray()
            expanded_matrix = adj_matrix[np.ix_(self.seq_idx_list[i], self.seq_idx_list[i])]
            expanded_matrix = torch.from_numpy(expanded_matrix).cuda()
            node_size = len(self.seq_idx_list[i])
            mask[i, :node_size, :node_size] *= expanded_matrix
        return mask

    def _get_dup_logit_mask(self, obs):
        batch_size = len(self.graphs)
        max_size = max([len(seq) for seq in self.seq_idx_list])
        mask = torch.ones((batch_size, max_size)).cuda()
        for i in range(batch_size):
            for j, vp in enumerate(self.seq_vp_list[i]):
                if vp == obs[i]['viewpoint']:
                    mask[i, j] = 0
                if self.args.no_temporal and self.seq_dup_vp[i][j]:
                    mask[i, j] = 0
        return mask

    def freeze_some_modules(self):
        self.fixed_modules = [self.vln_bert.vln_bert, self.vln_bert.position_encoder]
        for module in self.fixed_modules:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_some_modules(self):
        for module in self.fixed_modules:
            for param in module.parameters():
                param.requires_grad = True

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        batch_size = len(obs)

        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):  # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        rl_teacher_loss = 0.
        target_predict_loss = 0.

        # for backtrack
        visited = [set() for _ in range(batch_size)]

        base_position = np.stack([ob['position'] for ob in obs], axis=0)

        # global embedding
        hist_embeds = [self.vln_bert('history').expand(batch_size, -1, -1)]  # global embedding         # [b,1,d]
        hist_lens = [1 for _ in range(batch_size)]                                                  # [b]
        action_embeds = []
        action_lens = [0 for _ in range(batch_size)]

        self._init_graph(batch_size)
        # import ipdb;ipdb.set_trace()
        for t in range(self.args.max_action_len):
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens, ob_pos = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            graph_mask = self._get_connectivity_mask() if t > 0 and self.args.use_conn else None
            ob_pos = ob_pos - torch.from_numpy(base_position).cuda().unsqueeze(1)  # bs x num_obs x 3
            
            t_ob_inputs = {
                "mode": "observation",
                "ob_img_feats": ob_img_feats,
                "ob_ang_feats": ob_ang_feats,
                "ob_nav_types": ob_nav_types,
                "ob_cand_lens": ob_cand_lens,
                "ob_masks": ob_masks,
                'ob_position': ob_pos.float()
            }

            t_ob_embeds = self.vln_bert(**t_ob_inputs)          # [b, view, d]
            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,    # history before t step
                'hist_lens': hist_lens,
                'action_embeds': action_embeds,
                'action_lens': action_lens,
                'ob_embeds': t_ob_embeds,
                'ob_masks': ob_masks,
                'ob_nav_types': ob_nav_types,
                'return_states': True if self.feedback == 'sample' else False,
                'graph_mask': graph_mask,
            }

            t_outputs = self.vln_bert(**visual_inputs)
            logit = t_outputs[0]
            _hist_embeds = t_outputs[-1]
            max_hist_len = _hist_embeds.size(1)

            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)
            # mask out logits of the current position
            if t > 0:
                logit[:, 1:max_hist_len].masked_fill_(self._get_dup_logit_mask(obs) == 0, -float('inf'))
            # mask action embeds
            logit[:, 1:max_hist_len].fill_(-float('inf'))

            if train_ml is not None:
                # Supervised training
                target = self._teacher_action(obs, ended, max_hist_len)
                ml_loss += self.criterion(logit, target)

            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:  # default: skip
                bt_masks = torch.zeros(ob_nav_types.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                logit[:, max_hist_len:].masked_fill_(bt_masks, -float('inf'))

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)  # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit / self.args.rl_temperature, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id - max_hist_len == (ob_cand_lens[i] - 1) or next_id == self.args.ignoreid or \
                        ended[i]:  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            self._update_graph(obs, ended, cpu_a_t, max_hist_len)
            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len - 1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        if next_id >= max_hist_len:
                            prev_act_angle[i] = \
                                obs[i]['candidate'][next_id - max_hist_len]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                position = np.stack([ob['position'] for ob in obs], axis=0) - base_position  # bs x 3
                position = torch.from_numpy(position).cuda().float()

                t_hist_inputs = {
                    'mode': 'history',
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': t,
                    'position': position
                }

                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                hist_embeds.append(t_hist_embeds)

                t_action_inputs = {
                    'mode': 'action',
                    'prev_action_img_fts': hist_img_feats,      # [b,d]
                    'prev_action_ang_fts': prev_act_angle,      # [b,d]
                    'ob_step': t,
                    'position': position
                }

                t_action_embeds = self.vln_bert(**t_action_inputs)
                action_embeds.append(t_action_embeds.unsqueeze(1))

                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1
                        action_lens[i] += 1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, max_hist_len, traj=traj)
            obs = self.env._get_obs(t=t+1)

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:  # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens, ob_pos = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            t_ob_inputs = {
                "mode": "observation",
                "ob_img_feats": ob_img_feats,
                "ob_ang_feats": ob_ang_feats,
                "ob_nav_types": ob_nav_types,
                "ob_cand_lens": ob_cand_lens,
                "ob_masks": ob_masks,
            }

            t_ob_embeds = self.vln_bert(**t_ob_inputs)          # [b, view, d]
            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,    # history before t step
                'hist_lens': hist_lens,
                'action_embeds': action_embeds,
                'action_lens': action_lens,
                'ob_embeds': t_ob_embeds,
                'ob_masks': ob_masks,
                'ob_nav_types': ob_nav_types,
                'return_states': True 
            }
            temp_output = self.vln_bert(**visual_inputs)
            last_h_ = temp_output[1]

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss
                if self.feedback == 'sample':
                    rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                self.logs['critic_loss'].append(t_critic_loss.item())
                self.logs['policy_loss'].append(t_policy_loss.item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            if self.args.rl_teacher_only:
                rl_loss = rl_teacher_loss * self.args.rl_teacher_weight / batch_size
            else:
                rl_loss += rl_teacher_loss * self.args.rl_teacher_weight / batch_size
            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())  # critic loss + policy loss + entropy loss

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        """ Evaluate once on each instruction in the current environment """
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super().test(iters=iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()
        
    def train_speaker(self, n_iters, use_cache=False):
        self.vln_bert_optimizer.zero_grad()
        total_loss = 0
        for i in range(1, n_iters + 1):
            self.env.reset()
            loss = self.teacher_forcing(train_lm=True, use_cache=use_cache, cache_type=self.args.cache_type, cache_dir=self.args.cache_dir)
            loss = loss / n_iters
            total_loss += loss.item()
            # print(i, "/", n_iters, loss, total_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)
        self.vln_bert_optimizer.step()
        return total_loss
    
    def valid_speaker(self, infered_speech_path, wrapper=(lambda x: x)):
        # print("valid speaker ,, batch", self.env.batch_size)
        # print("valid_speaker",self.args.use_clip16 )
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        created = {}
        created_enc = {}
        gt = {}
        path = {}
        scan = {}
        total = self.env.size()
        for i in tqdm(wrapper(range(total // self.env.batch_size + 1))):  # Guarantee that all the data are processed
            obs = self.env.reset()

            insts = self.teacher_forcing(train_lm=False, use_cache=self.args.use_cache, cache_type=self.args.cache_type, cache_dir=self.args.cache_dir)
            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            gts = [ob['instruction'] for ob in obs]
            paths = [ob['gt_path'] for ob in obs]
            scans = [ob['scan'] for ob in obs]
            # print("path_ids", path_ids)
            if self.args.max_given_len > 0:
                givens = [ob['given'] for ob in obs]  
            for j, (path_id, inst) in enumerate(zip(path_ids, insts)):
                if path_id not in path2inst:
                    if self.args.use_clip16:
                        created_enc[path_id] = inst
                        shrinked = self.tokenizer.shrink(inst)  # Shrink the words
                        created[path_id] = self.tokenizer.decode_sentence(shrinked)
                        path2inst[path_id] = created[path_id] # bleu score 계산방식 수정 
                    else: #bert, t5
                        path2inst[path_id] = inst
                        # non_zero_index = np.argmax(inst != 0)
                        # trimmed_arr = inst[non_zero_index:]
                        # print("inst", inst, self.tokenizer.decode(path2inst[path_id]))
                        created[path_id] = self.tokenizer.decode(inst)
                        path2inst[path_id] = created[path_id]
                        # print("!!! decode result .. ", inst, created )
                    gt[path_id] = gts[j]
                    path[path_id] = paths[j]
                    scan[path_id] = scans[j]
                item = {
                    "scan" : scan[path_id],
                    "path_id": path_id, 
                    "created" : created[path_id],
                    # "created_enc" : created_enc[path_id],
                    "gt" : gt[path_id],
                    "path" : path[path_id],
                    # "path2inst": inst.tolist(), 
                    # "path2inst": path2inst[path_id].tolist(), 
                }
                if self.args.max_given_len > 0:
                    item["given"] = givens[j] 
                write_to_record_file(str(item), infered_speech_path)
        return path2inst
    
    def valid_speaker_for_vis(self, wrapper=(lambda x: x)):
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        total = self.env.size()
        for i in tqdm(wrapper(range(total // self.env.batch_size + 1))):  # Guarantee that all the data are processed
            obs = self.env.reset()

            insts = self.rollout(iters=None, vis_cap=True)
            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = self.tokenizer.shrink(inst)  # Shrink the words
        return path2inst
    
    def train_cont(self, n_iters):
        for i in range(1, n_iters + 1):
            self.env.reset()
            self.vln_bert_optimizer.zero_grad()
            loss = self.teacher_forcing(train_cont=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)
            self.vln_bert_optimizer.step()
            return loss.item()

    def cont_loss(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        # sim_loss = nce_loss.mean()
        return nce_loss

        
    def make_future_mask(
            self, size: int, dtype: torch.dtype, device: torch.device
        ) -> torch.Tensor:
        """
        Generate a mask for "future" positions. Masked positions will be negative
        infinity. This mask is critical for casual language modeling.
        """
        return torch.triu(
            torch.full((size, size), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )

    # def get_history_and_actions_bu(self, obs):
    #     use_cache = self.args.use_cache
    #     add_eot = self.args.eot_token

    #     print("Current cache size", len(self.action_cache))

    #     cached = [ob['cache_key'] in self.action_cache for ob in obs]

    #     if use_cache and all(cached):
    #         None
    #     else:
    #         print("CACHE MISS!!!!!!!!!!!!!!", cached)
    #         print([ob['cache_key'] for ob in obs])
    #         batch_size = len(obs)
    #         # hist_lens = [1 for _ in range(batch_size)]                                                  # [b]
    #         hist_lens = torch.ones(batch_size, dtype=torch.long).cuda()  # [b]
    #         # action_lens = [0 for _ in range(batch_size)]
    #         action_lens = torch.zeros(batch_size, dtype=torch.long).cuda()  # [b]


    #         hist_pano_img = []
    #         hist_pano_ang = []
    #         hist_img = []
    #         prev_act = []

    #         traj = [{
    #             'instr_id': ob['instr_id'],
    #             'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
    #         } for ob in obs]
    #         ended = np.array([False] * batch_size)
    #         for t in range(self.args.max_action_len):  
    #             if self.args.ob_type == 'pano':
    #                 ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens, ob_pos = self._cand_pano_feature_variable(obs)
    #                 ob_masks = length2mask(ob_lens).logical_not()

    #             elif self.args.ob_type == 'cand':
    #                 ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
    #                 ob_masks = length2mask(ob_cand_lens).logical_not()

    #             # time.sleep(1)
    #             # print_memory_usage("after "+ str(t)+" th action")

    #             target = self._teacher_action(obs, ended, 0)
    #             a_t = target

    #             cpu_a_t = a_t.cpu().numpy()
    #             for i, next_id in enumerate(cpu_a_t):
    #                 if next_id == (ob_cand_lens[i]-1) or next_id == self.args.ignoreid or ended[i]:    # The last action is <end>
    #                     cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

    #             # if ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len-1)):
    #             # DDP error: RuntimeError: Expected to mark a variable ready only once.
    #             # It seems that every output from DDP should be used in order to perform correctly
    #             hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
    #             prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
    #             for i, next_id in enumerate(cpu_a_t):
    #                 if next_id != -1:
    #                     prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
    #             prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

    #             hist_pano_img.append(hist_pano_img_feats) # [b,d] => [t, b, d]
    #             hist_pano_ang.append(hist_pano_ang_feats) # [b,d]
    #             hist_img.append(hist_img_feats)
    #             prev_act.append(prev_act_angle)

    #             for i, i_ended in enumerate(ended):
    #                 if not i_ended:
    #                     hist_lens[i] += 1
    #                     action_lens[i] += 1
    #             self.make_equiv_action(cpu_a_t, obs, 0, traj)
    #             obs = self.env._get_obs(t=t+1)
                
    #             ended[:] = np.logical_or(ended, (cpu_a_t == -1))
                
    #             # print_memory_usage("after "+ str(t)+" th action features")
            
                        
    #         # Add EOT token if add_eot is True and trajectory ended early
    #         if self.args.debug:
    #             print("add_eot", add_eot, "instr_ids", [(ob['instr_id'], ob['gt_path'], ended[i]) for i, ob in 
    #             enumerate(obs)])
    #             print()
    #             print()

                
    #         if add_eot:
    #             for idx, ob in enumerate(obs):
    #                 if ended[idx]: 
    #                     hist_pano_img.append(torch.zeros_like(hist_pano_img_feats).cuda())
    #                     hist_pano_ang.append(torch.zeros_like(hist_pano_ang_feats).cuda())
    #                     hist_img.append(torch.zeros_like(hist_img_feats).cuda())
    #                     prev_act.append(torch.zeros_like(prev_act_angle).cuda())
    #                     hist_lens[idx] += 1
    #                     action_lens[idx] += 1

    #         hist_pano_img_tensor = torch.stack(hist_pano_img, dim=0)
    #         hist_pano_img_tensor = hist_pano_img_tensor.permute(1, 0, 2, 3)  
    #         hist_pano_ang_tensor = torch.stack(hist_pano_ang, dim=0).permute(1, 0, 2, 3)  
    #         hist_img_tensor = torch.stack(hist_img, dim=0).permute(1, 0, 2)  
    #         prev_act_tensor = torch.stack(prev_act, dim=0).permute(1, 0, 2)  

    #         for idx, ob in enumerate(obs):
    #             cache_key = ob['cache_key']
    #             self.action_cache[cache_key] = {
    #                 'hist_pano_img_tensor': hist_pano_img_tensor[idx],
    #                 'hist_pano_ang_tensor': hist_pano_ang_tensor[idx],
    #                 'hist_img_tensor': hist_img_tensor[idx],
    #                 'prev_act_tensor': prev_act_tensor[idx],
    #                 'hist_len': hist_lens[idx],
    #                 'action_len': action_lens[idx]
    #             }

    def load_cache_keys(self):
        cache_dir = self.args.cache_dir
        if not os.path.exists(cache_dir):
            return
        for directory in os.listdir(cache_dir):
            dir_path = os.path.join(cache_dir, directory)
            if not os.path.isdir(dir_path):
                continue
            for file in os.listdir(dir_path):
                loaded = torch.load(os.path.join(dir_path, file))
                # print("Adding keys from disk", loaded['cache_key'])
                self.cache_keys.add(loaded['cache_key'])

    def get_history_and_actions(self, obs, use_cache=False, cache_type='gpu', cache_dir='../cache'):
        add_eot = self.args.eot_token
        if cache_type == 'disk':
            import hashlib

            def get_cache_path(key):
                first_vp = key.split(',')[0]
                hash_str = hashlib.md5(key.encode()).hexdigest()
                hash_str = hash_str[:30]
                cache_subdir = os.path.join(cache_dir, first_vp)
                os.makedirs(cache_subdir, exist_ok=True)
                return os.path.join(cache_subdir, f"{hash_str}")
            
            def update_cache(key, new_item):
                cache_path = get_cache_path(key)
                optimized_data = {'cache_key': key}
                for k, v in new_item.items():
                    if torch.is_tensor(v):
                        optimized_data[k] = v.clone().detach().contiguous()
                    else:
                        optimized_data[k] = v
                torch.save(optimized_data, cache_path)
                self.cache_keys.add(key)
            
            def load_cache(key):
                cache_path = get_cache_path(key)
                return torch.load(cache_path)

        cache_hits = []
        for ob in obs:
            cache_key = ob['cache_key']
            if cache_type == 'gpu':
                hit = cache_key in self.action_cache
            else:
                hit = cache_key in self.cache_keys
            cache_hits.append(hit)
        
        cache_source = {}
        if use_cache and all(cache_hits):
            if cache_type == 'gpu':
                cache_source = self.action_cache
            elif cache_type == 'disk':
                for ob in obs:
                    cache_key = ob['cache_key']
                    cache_source[cache_key] = load_cache(cache_key)
            else:
                raise ValueError("Invalid cache type")

        else:
            if use_cache:
                cache_size = len(self.cache_keys) if cache_type == 'disk' else len(self.action_cache)
                print(f"use_cache {use_cache} hits {sum(cache_hits)/len(cache_hits)} .. cache size {cache_size}")

            batch_size = len(obs)
            hist_lens = torch.ones(batch_size, dtype=torch.long).cuda()
            action_lens = torch.zeros(batch_size, dtype=torch.long).cuda()

            hist_pano_img = []
            hist_pano_ang = []
            hist_img = []
            prev_act = []

            traj = [{
                'instr_id': ob['instr_id'],
                'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            } for ob in obs]
            ended = np.array([False] * batch_size)
            for t in range(self.args.max_action_len):  
                if self.args.ob_type == 'pano':
                    ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens, ob_pos = self._cand_pano_feature_variable(obs)
                    ob_masks = length2mask(ob_lens).logical_not()
                elif self.args.ob_type == 'cand':
                    ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                    ob_masks = length2mask(ob_cand_lens).logical_not()

                target = self._teacher_action(obs, ended, 0)
                a_t = target

                cpu_a_t = a_t.cpu().numpy()
                for i, next_id in enumerate(cpu_a_t):
                    if next_id == (ob_cand_lens[i]-1) or next_id == self.args.ignoreid or ended[i]:    
                        cpu_a_t[i] = -1             

                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                hist_pano_img.append(hist_pano_img_feats)
                hist_pano_ang.append(hist_pano_ang_feats)
                hist_img.append(hist_img_feats)
                prev_act.append(prev_act_angle)

                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1
                        action_lens[i] += 1
                self.make_equiv_action(cpu_a_t, obs, 0, traj)
                obs = self.env._get_obs(t=t+1)
                
                ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            hist_pano_img_tensor = torch.stack(hist_pano_img, dim=0)
            hist_pano_img_tensor = hist_pano_img_tensor.permute(1, 0, 2, 3)  
            hist_pano_ang_tensor = torch.stack(hist_pano_ang, dim=0).permute(1, 0, 2, 3)  
            hist_img_tensor = torch.stack(hist_img, dim=0).permute(1, 0, 2)  
            prev_act_tensor = torch.stack(prev_act, dim=0).permute(1, 0, 2)  

            for idx, ob in enumerate(obs):
                cache_key = ob['cache_key']
                cache_data = {
                    'hist_pano_img_tensor': hist_pano_img_tensor[idx],
                    'hist_pano_ang_tensor': hist_pano_ang_tensor[idx],
                    'hist_img_tensor': hist_img_tensor[idx],
                    'prev_act_tensor': prev_act_tensor[idx],
                    'hist_len': hist_lens[idx],
                    'action_len': action_lens[idx]
                }
                
                cache_source[cache_key] = cache_data
                if use_cache:
                    if cache_type == 'disk':
                        update_cache(cache_key, cache_data)
                    else:
                        self.action_cache[cache_key] = cache_data

        # Find max sequence length across all samples
        max_seq_len = max(len(cache_source[ob['cache_key']]['hist_pano_img_tensor']) 
                         for ob in obs)
        
        # Process and pad each item to max length
        processed_tensors = []
        for idx, ob in enumerate(obs):
            cache_data = cache_source[ob['cache_key']]
            hist_pano_img = cache_data['hist_pano_img_tensor']
            hist_pano_ang = cache_data['hist_pano_ang_tensor'] 
            hist_img = cache_data['hist_img_tensor']
            prev_act = cache_data['prev_act_tensor']
            hist_len_item = cache_data['hist_len']
            action_len_item = cache_data['action_len']

            # Calculate padding needed
            pad_len = max_seq_len - len(hist_pano_img)
            
            # Pad tensors to max length
            if pad_len > 0:
                hist_pano_img = torch.cat([hist_pano_img, torch.zeros_like(hist_pano_img[:1]).repeat(pad_len,1,1)], dim=0)
                hist_pano_ang = torch.cat([hist_pano_ang, torch.zeros_like(hist_pano_ang[:1]).repeat(pad_len,1,1)], dim=0)
                hist_img = torch.cat([hist_img, torch.zeros_like(hist_img[:1]).repeat(pad_len,1)], dim=0)
                prev_act = torch.cat([prev_act, torch.zeros_like(prev_act[:1]).repeat(pad_len,1)], dim=0)

            processed_tensors.append({
                'hist_pano_img': hist_pano_img,
                'hist_pano_ang': hist_pano_ang,
                'hist_img': hist_img,
                'prev_act': prev_act,
                'hist_len': hist_len_item,
                'action_len': action_len_item
            })

        # Now stack all processed tensors which should be same size
        hist_pano_img_tensor = torch.stack([item['hist_pano_img'] for item in processed_tensors], dim=0)
        hist_pano_ang_tensor = torch.stack([item['hist_pano_ang'] for item in processed_tensors], dim=0)
        hist_img_tensor = torch.stack([item['hist_img'] for item in processed_tensors], dim=0)
        prev_act_tensor = torch.stack([item['prev_act'] for item in processed_tensors], dim=0)
        hist_len = torch.tensor([item['hist_len'] for item in processed_tensors]).cuda()
        action_len = torch.tensor([item['action_len'] for item in processed_tensors]).cuda()

        # Permute dimensions
        hist_pano_img_tensor = hist_pano_img_tensor.permute(1, 0, 2, 3)
        hist_pano_ang_tensor = hist_pano_ang_tensor.permute(1, 0, 2, 3)
        hist_img_tensor = hist_img_tensor.permute(1, 0, 2)
        prev_act_tensor = prev_act_tensor.permute(1, 0, 2)

        t_hist_inputs_list = []
        t_action_inputs_list = []
        for t, t_hist_pano_img_tensor in enumerate(hist_pano_img_tensor):
            t_hist_inputs = {
                'mode': 'history',
                'hist_pano_img_feats': t_hist_pano_img_tensor,
                'hist_pano_ang_feats': hist_pano_ang_tensor[t],
                'ob_step': t,
            }
            t_hist_inputs_list.append(t_hist_inputs)

            t_action_inputs = {
                'mode': 'action',
                'prev_action_img_fts': hist_img_tensor[t],      # [b,d]
                'prev_action_ang_fts': prev_act_tensor[t],      # [b,d]
                'ob_step': t
            }
            t_action_inputs_list.append(t_action_inputs)

        return t_hist_inputs_list, t_action_inputs_list, hist_len, action_len 

    def teacher_forcing(self, train_lm=False, train_cont=False, use_cache=False, cache_type='gpu', cache_dir='../cache'):
        if train_lm or train_cont:
            self.vln_bert.train()
        else:
            self.vln_bert.eval()
        obs = self.env._get_obs(t=0)
        batch_size = len(obs)

        hist_embeds = [self.vln_bert('history').expand(batch_size, -1, -1)]  # global embedding         # [b,1,d]
        hist_lens = [1 for _ in range(batch_size)]                                                  # [b]
        action_embeds = []
        action_lens = [0 for _ in range(batch_size)]

        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        t_hist_inputs_list, t_action_inputs_list, hist_lens, action_lens  = self.get_history_and_actions(obs, use_cache, cache_type, cache_dir)
        hist_embeds = [self.vln_bert('history').expand(batch_size, -1, -1)]  # global embedding         # [b,1,d]
        action_embeds = []
        for t, action_input in enumerate(t_action_inputs_list):
            t_hist_inputs = t_hist_inputs_list[t]
            t_action_inputs =  t_action_inputs_list[t]
            t_hist_embeds = self.vln_bert(**t_hist_inputs)
            hist_embeds.append(t_hist_embeds)
            t_action_embeds = self.vln_bert(**t_action_inputs)
            action_embeds.append(t_action_embeds.unsqueeze(1))
        
        if train_lm:
            # language embedding
            txt_ids, txt_masks, txt_lens = self._language_variable(obs)
            gt_txt = txt_ids
            future_mask = self.make_future_mask(
                txt_ids.shape[1], torch.float, txt_ids.device
            )
            language_inputs = {
                'mode': 'language',
                'txt_ids': txt_ids,
                'txt_masks': txt_masks,
                'future_mask': future_mask
            }
            txt_embeds = self.vln_bert(**language_inputs)           # [b,len,d=768]

            caption_input = {
                'mode': 'visual',
                'hist_embeds': hist_embeds,
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_lens': hist_lens,
                'action_embeds': action_embeds,
                'action_lens': action_lens,
                'is_train_caption': True,
                'future_mask': future_mask
            }

            prediction_scores = self.vln_bert(**caption_input)   # [b,l,n_vocab]
            bs, l, n_vocab = prediction_scores.shape
            if  self.args.max_given_len == 0:
                lm_loss = F.cross_entropy(
                    prediction_scores[:, :-1].contiguous().view(-1, n_vocab),
                    gt_txt[:,1:].contiguous().view(-1),
                    ignore_index=0, reduction='mean')
            elif self.args.train_given_logits:
                lm_loss = F.cross_entropy(
                    prediction_scores[:, :-1].contiguous().view(-1, n_vocab),
                    gt_txt[:,1:].contiguous().view(-1),
                    ignore_index=0, reduction='mean')
            else: # given text
                lm_loss = F.cross_entropy(
                    prediction_scores[:, self.args.max_given_len:-1].contiguous().view(-1, n_vocab),
                    gt_txt[:,1+self.args.max_given_len:].contiguous().view(-1),
                    ignore_index=0, reduction='mean')
                
            # print("lm_loss", prediction_scores)
            # print("lm_loss", lm_loss)

            return lm_loss
        
        elif train_cont:
            txt_ids, txt_masks, txt_lens = self._language_variable(obs)
            language_inputs = {
                'mode': 'language',
                'txt_ids': txt_ids,
                'txt_masks': txt_masks,
            }
            txt_embeds = self.vln_bert(**language_inputs)
            contrastive_input = {
                'mode': 'visual',
                'hist_embeds': hist_embeds,
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_lens': hist_lens,
                'action_embeds': action_embeds,
                'action_lens': action_lens,
                'is_train_contrastive': True
            }
            cont_sim = self.vln_bert(**contrastive_input)
            cont_loss = (self.cont_loss(cont_sim) + self.cont_loss(cont_sim.T)) / 2.0
            return cont_loss.mean()
        
        max_decode = self.args.max_instr_len
        bs = len(obs)
        ended = torch.zeros(bs, dtype=torch.bool).cuda()


        if self.args.max_given_len  == 0:
            words = torch.ones(bs, 1, dtype=torch.long) * self.cls_token_id     # [b,1]
            words = words.cuda()
        else:
            que_lengths = [len(ob['given_encoding']) for ob in obs]
            max_que_lengths = max(que_lengths)
            que_tensor = np.zeros((len(obs), max_que_lengths))
            GIVEN_LENGTH = self.args.max_given_len 
            for i, ob in enumerate(obs):
                que_tensor[i, :GIVEN_LENGTH] = ob['given_encoding']
            que_tensor = torch.from_numpy(que_tensor)
            words = que_tensor.long().cuda()

        for i in range(max_decode):
            future_mask = self.make_future_mask(words.shape[1], hist_embeds[0].dtype, words.device)
            caption_lengths = (words != 0).sum(-1)
            ones = torch.ones_like(words)
            caption_mask = caption_lengths.unsqueeze(1) < ones.cumsum(dim=1)
            language_inputs = {
                'mode': 'language',
                'txt_ids': words,
                'txt_masks': caption_mask,
                'future_mask': future_mask
            }
            txt_embeds = self.vln_bert(**language_inputs)           # [b,len,d=768]
            caption_input = {
                'mode': 'visual',
                'hist_embeds': hist_embeds,
                'txt_embeds': txt_embeds,
                'txt_masks': caption_mask,
                'hist_lens': hist_lens,
                'action_embeds': action_embeds,
                'action_lens': action_lens,
                'is_train_caption': True,
                'future_mask': future_mask
            }

            logits = self.vln_bert(**caption_input)   # [b,l,n_vocab]
            logits = logits[:,-1,:]
            values, word = logits.max(-1)
            word[ended] = self.pad_token_id
            words = torch.cat([words, word.unsqueeze(-1)], dim=-1)
            ended = torch.logical_or(ended, word == self.sep_token_id)
            if ended.all():
                break
        
        if self.args.max_given_len  == 0: 
            return words.cpu().numpy()  
        else:
            answers = torch.stack([row[max_que_lengths:] for row in words])
            return answers.cpu().numpy()
        
    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0
            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            # print(self.rank, iter, self.loss)
            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if self.args.aug is None:
                print_progress(iter, n_iters + 1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path, states=None):
        ''' Loads parameters (but not training state) '''
        if states is None:
            states = torch.load(path, weights_only=True)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())

            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            if model_keys != load_keys:
                # print("NOTICE: DIFFERENT KEYS IN THE CHECKPOINT")
                # # Log missing and unexpected keys
                # missing_keys = model_keys - load_keys
                # unexpected_keys = load_keys - model_keys
                # print("Missing keys in checkpoint:", missing_keys)
                # print("Unexpected keys in checkpoint:", unexpected_keys)

                # # Filter out unexpected keys
                # # state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])

        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
