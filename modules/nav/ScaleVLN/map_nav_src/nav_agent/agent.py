import sys
import numpy as np
import random
import math

import torch
import torch.nn as nn

from sv_utils.distributed import is_default_gpu
from sv_utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent, BaseAgent

from sv_models.graph_utils import GraphMap
from sv_models.model import VLNBert, Critic
from sv_models.ops import pad_tensors_wgrad
import torch.nn.functional as F
class GMapNavAgent(Seq2SeqAgent):
    
    def _build_model(self):
        print(torch.__version__)
        print(torch.version.cuda)
        print(torch.backends.cudnn.version())
        print(torch.cuda.is_available())
        
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}

    def _language_variable(self, instr_encoding_list):
        seq_lengths = [len(instr_encoding) for instr_encoding in instr_encoding_list]
        
        seq_tensor = np.zeros((len(instr_encoding_list), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(instr_encoding_list), max(seq_lengths)), dtype=np.bool)
        for i, instr_encoding in enumerate(instr_encoding_list):
            seq_tensor[i, :seq_lengths[i]] = instr_encoding
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids,
        }

    def _update_gmap(self, gmaps, obs, ended, action_idx, avg_pano_embeds, pano_inputs, pano_embeds, steps=None):
        for i, gmap in enumerate(gmaps):
            if not ended[i]:
                # update visited node
                gmap.node_step_ids[obs[i]['viewpoint']] = (steps[i] if steps is not None else 0) + action_idx + 1
                i_vp = obs[i]['viewpoint']
                gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                # update unvisited nodes
                for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                    if not gmap.graph.visited(i_cand_vp):
                        gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

    def _nav_probs(self, nav_inputs):
        nav_outs = self.vln_bert('navigation', nav_inputs)
        if self.args.fusion == 'local':
            nav_logits = nav_outs['local_logits']
            nav_vpids = nav_inputs['vp_cand_vpids']
        elif self.args.fusion == 'global':
            nav_logits = nav_outs['global_logits']
            nav_vpids = nav_inputs['gmap_vpids']
        else:
            nav_logits = nav_outs['fused_logits']
            nav_vpids = nav_inputs['gmap_vpids']

        nav_probs = torch.softmax(nav_logits, 1)        
        return nav_probs, nav_vpids, nav_logits, nav_outs

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i], 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def _teacher_action_r4r(
        self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t], "instr_id: %s, viewpoint: %s, gt_path: %s, t: %s" % (ob['instr_id'], ob['viewpoint'], ob['gt_path'], t)
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived 
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()
    
    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                # print("traj[i]['path']", traj[i]['path'], ob['viewpoint'], action)
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                # viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]

                try:  
                    viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                except Exception as e:
                    print(f"Error at i: {i}, scan: {ob['scan']}, prev_vp: {prev_vp}, action: {action}, scanvp_cands: {self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)]}")
                    print(f"Error: {e}")
                    print(f"obs : {ob['instr_id']}")

                
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])
    def _update_scanvp_cands(self, obs):
        for idx, ob in enumerate(obs):
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    def decide_action(self, nav_logits, nav_probs, nav_vpids, nav_inputs, obs, ended, batch_size, t, nav_targets=None):
        # Determinate the next navigation viewpoint
        if self.feedback == 'teacher':
            a_t = nav_targets                 # teacher forcing
        elif self.feedback == 'argmax':
            _, a_t = nav_logits.max(1)        # student forcing - argmax
            a_t = a_t.detach() 
        elif self.feedback == 'sample':
            c = torch.distributions.Categorical(nav_probs)
            self.logs['entropy'].append(c.entropy().sum().item())            # For log
            a_t = c.sample().detach()
            entropy = c.entropy() 
        elif self.feedback == 'expl_sample':
            _, a_t = nav_probs.max(1)
            rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
            if self.args.fusion == 'local':
                cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
            else:
                cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
            for i in range(batch_size):
                if rand_explores[i]:
                    cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                    a_t[i] = np.random.choice(cand_a_t)
        else:
            print(self.feedback)
            sys.exit('Invalid feedback option')

        # Determine stop actions
        if self.feedback == 'teacher' or self.feedback == 'sample': # in training
            # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
            a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
        else:
            a_t_stop = a_t == 0

        # Prepare environment action
        cpu_a_t = []  
        just_ended = np.array([False] * batch_size)
        for i in range(batch_size):
            if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                cpu_a_t.append(None)
                just_ended[i] = True
            else:
                cpu_a_t.append(nav_vpids[i][a_t[i]])   
                
        return a_t, a_t_stop, cpu_a_t, just_ended, entropy if self.feedback == 'sample' else None
    
    def _process_navigation_step(self, obs, gmaps, ended, nav_idx, language_inputs, txt_embeds, steps=None):
        """Process a single navigation step"""
        pano_inputs = self._panorama_feature_variable(obs)
        pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
        avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                          torch.sum(pano_masks, 1, keepdim=True)
        
        nav_vp_variable = self._nav_vp_variable(
            obs, gmaps, pano_embeds, pano_inputs['cand_vpids'], 
            pano_inputs['view_lens'], pano_inputs['nav_types'],
        )
        self._update_gmap(gmaps, obs, ended, nav_idx, avg_pano_embeds, pano_inputs, pano_embeds, steps)
        
        nav_inputs = self._nav_gmap_variable(obs, gmaps)
        nav_inputs.update(nav_vp_variable)
        nav_inputs.update({'txt_embeds': txt_embeds, 'txt_masks': language_inputs['txt_masks']}) 
        
        return nav_inputs, pano_inputs

    def _update_stop_node(self, obs, gmaps, ended, just_ended, traj, batch_size):
        """Update trajectory and graph after action execution"""
        for i in range(batch_size):
            if (not ended[i]) and just_ended[i]:
                stop_node, stop_score = None, {'stop': -float('inf')}
                for k, v in gmaps[i].node_stop_scores.items():
                    if v['stop'] > stop_score['stop']:
                        stop_score = v
                        stop_node = k
                if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                    traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                if self.args.detailed_output:
                    for k, v in gmaps[i].node_stop_scores.items():
                        traj[i]['details'][k] = {
                            'stop_prob': float(v['stop']),
                        }

    def _update_graph_structure(self, obs, gmaps, ended):
        # new observation and update graph
        self._update_scanvp_cands(obs)
        for i, ob in enumerate(obs):
            if not ended[i]:
                gmaps[i].update_graph(ob)
    
    def _update_stop_scores(self, nav_probs, gmaps, obs, ended):
        for i, gmap in enumerate(gmaps):
            if not ended[i]:
                i_vp = obs[i]['viewpoint']
                gmap.node_stop_scores[i_vp] = {
                    'stop': nav_probs[i, 0].data.item(),
                }

    def _initialize_graph(self, obs):
        return [GraphMap(ob['viewpoint']) for ob in obs]

    def _set_instruction(self, instr_encoding_list):
        language_inputs = self._language_variable(instr_encoding_list)
        txt_embeds = self.vln_bert('language', language_inputs)
        return language_inputs, txt_embeds
    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True, train_wta=False):
        if reset:  # Reset env
            self.env.reset(train=train_ml is not None)
        obs = self.env._get_obs()
        batch_size = len(obs)

        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # build graph: keep the start viewpoint
        gmaps = self._initialize_graph(obs)
        self._update_graph_structure(obs, gmaps, ended)
        instr_encoding_list = [ob['instr_encoding'] for ob in obs]
        language_inputs, txt_embeds = self._set_instruction(instr_encoding_list)



        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'wta_label': [],
            'wta_predict': [],
        } for ob in obs]



        ml_loss = 0.

        for nav_idx in range(self.args.max_action_len):
            # if nav_idx == 10:
            #     raise Exception("stop")
            nav_inputs, pano_inputs = self._process_navigation_step(obs, gmaps, ended, nav_idx, language_inputs, txt_embeds)
            nav_probs, nav_vpids, nav_logits, nav_outs = self._nav_probs(nav_inputs)

            nav_targets = None
            if train_ml is not None:
                # Supervised training
                nav_targets = self._teacher_action_r4r(
                    obs, nav_vpids, ended, 
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                    imitation_learning=(self.feedback=='teacher'), t=nav_idx, traj=traj
                )
                current_loss = self.criterion(nav_logits, nav_targets)
                
                if torch.isfinite(current_loss):
                    ml_loss += current_loss
                else:
                    print(f"Warning: Loss is {current_loss}, ignoring this step. nav_idx: {nav_idx}")
                    print("nav_logits", nav_logits)
                    print("nav_targets", nav_targets)
                    print("gmap masks", nav_inputs['gmap_visited_masks'])
                    # raise Exception("Loss is not finite")
                    # continue

            
            if train_ml and train_wta:
                self.wta_loss_batch += self._teacher_forcing_wta(obs, nav_outs, nav_targets, traj)

            self._update_stop_scores(nav_probs, gmaps, obs, ended)
            a_t, a_t_stop, cpu_a_t, just_ended, entropy = self.decide_action(nav_logits, nav_probs, nav_vpids, nav_inputs, obs, ended, batch_size, nav_idx, nav_targets)
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            obs = self.env._get_obs()
            self._update_stop_node(obs, gmaps, ended, just_ended, traj, batch_size)
            self._update_graph_structure(obs, gmaps, ended)
            ended = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())
        
        if train_ml and train_wta:
            self.wta_loss += self.wta_loss_batch
            self.wta_loss_batch = 0
            self.logs['wta_loss'].append(self.wta_loss.item())

        return traj

class GMapNavAgnetWta(GMapNavAgent):
    def _build_model(self, tok=None):
        print(torch.__version__)
        print(torch.version.cuda)
        print(torch.backends.cudnn.version())
        print(torch.cuda.is_available())

        self.vln_bert = VLNBert(self.args, True).cuda() # set wta module true 
        self.critic = Critic(self.args).cuda()
        self.scanvp_cands = {}
        self.tok = tok

        # self.question_weight = 1.0
        # if self.args.question_weight is not None:
        #     self.question_weight = self.args.question_weight
            

        # wta_weights = torch.tensor([1.0, 30.0]).cuda()  # Weights for class 0 and class 1
        # self.criterion_question = nn.CrossEntropyLoss(weight=wta_weights, ignore_index=self.args.ignoreid, reduction='sum')
        # self.wta_module = None 
    

    def decide_wta(self, ended, nav_inputs):
        nav_outs = self.vln_bert('navigation', nav_inputs)
        wta_logits = nav_outs['question_logits']
        wta_probs = F.softmax(wta_logits, dim=1)
        _, ask = wta_probs.max(1)
        ask = ask.cpu().numpy()  # Move the tensor to CPU and convert to NumPy array
        ask = np.logical_and(np.logical_not(ended), ask) 
        return ask

    
class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        self.env.reset()
        obs = self.env._get_obs()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        self.steps = random.sample(range(-11,1), len(obs))
        ended = [False] * len(obs)
        # for t in range(30):
        for t in range(50):
            actions = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append((0, 0, 0)) # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] < 0:
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else:
                    actions.append((0, 1, 0)) # turn right until we can go forward
            obs = self.env.step(actions)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
        return traj