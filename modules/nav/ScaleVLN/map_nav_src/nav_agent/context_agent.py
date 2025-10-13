import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .agent import GMapNavAgent

from sv_models.graph_utils import GraphMap
from sv_models.vlnbert_init import get_tokenizer


class GMapWithContextAgent(GMapNavAgent):
    def __init__(self, args, env, rank=0):
        super().__init__(args, env, rank)
        self.tok = get_tokenizer(args)
    
    def _get_next_action_with_ref_path(self, obs, ref_paths, t, ended):
        a = np.zeros(len(obs), dtype=np.object_)
        for i, ob in enumerate(obs):
            if t < len(ref_paths[i]) - 1:
                a[i] = ref_paths[i][t+1]
            else:
                a[i] = None
                ended[i] = True

        return a, ended
    
    def init_graph_with_path_trajectory(self, obs):
        historys = [ob['context'] for ob in obs]
        scanIds = [ob['scan'] for ob in obs]
        initial_headings = [3.141592653589793 for history in historys]
        nav_historys = [history['nav_history'] for history in historys]
        full_dialogs = [history['_full_dialog'] for history in historys]
        nav_historys = [nav_history[-self.args.maximum_navigation_history_length:] for nav_history in nav_historys]
        initial_viewpoints = [history[0] for history in nav_historys]
        max_nav_history_len = max([len(nav_history) for nav_history in nav_historys])

        self.env.env.newEpisodes(scanIds, initial_viewpoints, initial_headings)
        obs = self.env._get_obs()
        gmaps = [GraphMap(initial_viewpoints[i]) for i in range(len(obs))]
        
        traj_history = [{ 'instr_id': ob['instr_id'], 'path': [[ob['viewpoint']]] } for ob in obs]
        ended = np.array([False] * len(obs))
        steps = [0] * len(obs)

        for t in range(max_nav_history_len-1):
            self._update_graph_structure(obs, gmaps, ended)
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)
            
            for i, gmap in enumerate(gmaps):
                if ended[i]:
                    continue

                gmap.node_step_ids[obs[i]['viewpoint']] =  steps[i] + 1
                steps[i] += 1

                gmap.update_node_embed(obs[i]['viewpoint'], avg_pano_embeds[i], rewrite=True)
                for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                    if not gmap.graph.visited(i_cand_vp):
                        gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])
            
            next_viewpoints, ended = self._get_next_action_with_ref_path(
                obs, nav_historys, t, ended
            )
            
            self.make_equiv_action(next_viewpoints, gmaps, obs, traj_history)
            obs = self.env._get_obs()
        
        traj = [{
                'instr_id': ob['instr_id'],
                'path': [[ob['viewpoint']]],
                'details': {},
                'traj_history': traj_history[i]['path'],
            } for i, ob in enumerate(obs)]
        
        # print("history length", [len(traj_history[i]['path']) for i in range(len(traj_history))])
        return {
            'gmaps': gmaps,
            'traj': traj,
            'obs': obs,
            'steps': steps,
        }
    

    def _language_variable_add_answer(self, obs, answer_list, append_question=False, question_text_list = [], append_dialog_history=False, dialog_history=[]):
        instructions = [self.tok.encode(text)[1:] if text else [] for text in answer_list]
        if append_question:
            for i, text in enumerate(question_text_list):
                if text:
                    instructions[i] = self.tok.encode(text)[1:] + instructions[i]

        if append_dialog_history:
            for i, history in enumerate(dialog_history):
                encoded_history_batch = []
                for history_item in history:
                    encoded_history_batch += self.tok.encode(history_item["question"])[1:]
                    encoded_history_batch += self.tok.encode(history_item["answer"])[1:]
                instructions[i]  = encoded_history_batch + instructions[i] 

        for i, ob in enumerate(obs):
            target_encoded = self.tok.encode(ob["target_only_instruction"])
            instructions[i] = [target_encoded[0]] + instructions[i]  + target_encoded[1:]

        total_lengths = [min(len(inst), self.args.max_instr_len) for inst in instructions]
        seq_tensor = np.zeros((len(obs), max(total_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(total_lengths)), dtype=bool)
            
        for i, instr in enumerate(instructions):
            seq_tensor[i, :total_lengths[i]] = instr[-total_lengths[i]:]
            mask[i, :total_lengths[i]] = True

        decoded = [self.tok.decode(seq) for seq in seq_tensor]
        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 
            'txt_masks': mask, 
            'decoded': decoded
        }
    
    def _get_dialog_maps(self, full_dialogs, max_nav_history_len):
        # Get dialog maps for current navigation index
        # dialog_maps[nav_idx][i] : qa item if dialog happend for i-th batch item at nav_idx

        dialog_maps = []
        for nav_idx in range(max_nav_history_len):
            answer_list = []
            question_list = []
            for full_dialog in full_dialogs:
                qa = [item for item in full_dialog if item['nav_idx'] == nav_idx]
                answer_list.append(qa[0]['a'] if len(qa) > 0 else None)
                question_list.append(qa[0]['q'] if len(qa) > 0 else None)
            dialog_maps.append((answer_list, question_list))
        return dialog_maps
    
    def update_dialog(self, language_inputs, txt_embeds, dialog_inputs, dialog_txt_embeds, to_ask_idices):
        """
        Update language inputs and embeddings with dialog information for samples that have answers.
        This function handles padding and merging of language and dialog inputs.
        """
        # Find max length across both language and dialog inputs
        max_len = max(
            language_inputs['txt_ids'].shape[1],
            dialog_inputs['txt_ids'].shape[1]
        )

        # Pad language inputs to max length if needed
        if language_inputs['txt_ids'].shape[1] < max_len:
            pad_len = max_len - language_inputs['txt_ids'].shape[1]
            language_inputs['txt_ids'] = torch.cat([
                language_inputs['txt_ids'],
                torch.zeros(language_inputs['txt_ids'].shape[0], pad_len, dtype=language_inputs['txt_ids'].dtype).cuda()
            ], dim=1)
            language_inputs['txt_masks'] = torch.cat([
                language_inputs['txt_masks'],
                torch.zeros(language_inputs['txt_masks'].shape[0], pad_len, dtype=language_inputs['txt_masks'].dtype).cuda()
            ], dim=1)
            txt_embeds = torch.cat([
                txt_embeds,
                torch.zeros(txt_embeds.shape[0], pad_len, txt_embeds.shape[2]).cuda()
            ], dim=1)

        # Pad dialog inputs to max length if needed  
        if dialog_inputs['txt_ids'].shape[1] < max_len:
            pad_len = max_len - dialog_inputs['txt_ids'].shape[1]
            dialog_inputs['txt_ids'] = torch.cat([
                dialog_inputs['txt_ids'],
                torch.zeros(dialog_inputs['txt_ids'].shape[0], pad_len, dtype=dialog_inputs['txt_ids'].dtype).cuda()
            ], dim=1)
            dialog_inputs['txt_masks'] = torch.cat([
                dialog_inputs['txt_masks'], 
                torch.zeros(dialog_inputs['txt_masks'].shape[0], pad_len, dtype=dialog_inputs['txt_masks'].dtype).cuda()
            ], dim=1)
            dialog_txt_embeds = torch.cat([
                dialog_txt_embeds,
                torch.zeros(dialog_txt_embeds.shape[0], pad_len, dialog_txt_embeds.shape[2]).cuda()
            ], dim=1)

        # Update embeddings and inputs for samples with answers
        for i in to_ask_idices:
            txt_embeds[i] = dialog_txt_embeds[i]
            language_inputs['txt_ids'][i] = dialog_inputs['txt_ids'][i]
            language_inputs['txt_masks'][i] = dialog_inputs['txt_masks'][i]
    
        return language_inputs, txt_embeds

    # def _print_graph_structure(self, gmaps, obs):


    #     for i, ob in enumerate(obs):
    #         viewpoint = gmaps[i].node_positions.keys()
    #         candidates = [c['viewpointId'] for c in ob['candidate']]
    #         print(f"{i} : instr_id : {ob['instr_id']}, viewpoint : {ob['viewpoint']}, candidate : {len(candidates)} ({candidates})")
    #         print(f"step : {gmaps[i].node_step_ids}")
    #         print("-- graph: ", len(viewpoint), viewpoint)
            
    #     print()
    #     print()

    def init_graph_with_full_dialog(self, obs):
        historys = [ob['context'] for ob in obs]
        scanIds = [ob['scan'] for ob in obs]
        initial_headings = [3.14 for history in historys]
        nav_historys = [history['nav_history'] for history in historys]
        full_dialogs = [history['_full_dialog'] for history in historys]
        initial_viewpoints = [history[0] for history in nav_historys]
        max_nav_history_len = max([len(nav_history) for nav_history in nav_historys])

        self.env.env.newEpisodes(scanIds, initial_viewpoints, initial_headings)
        obs = self.env._get_obs()
        gmaps = [GraphMap(initial_viewpoints[i]) for i in range(len(obs))]
        
        traj_history = [{ 'instr_id': ob['instr_id'], 'path': [[ob['viewpoint']]] } for ob in obs]
        ended = np.array([False] * len(obs))
        steps = [0] * len(obs)
        instr_encoding_list = [ob['target_only_instr_encoding'] for ob in obs]
        language_inputs = self._language_variable(instr_encoding_list)
        txt_embeds = self.vln_bert('language', language_inputs)

        dialog_maps = self._get_dialog_maps(full_dialogs, max_nav_history_len)

        for nav_idx in range(max_nav_history_len-1):
            self._update_graph_structure(obs, gmaps, ended)
        
            ## update dialog
            answer_list, question_list = dialog_maps[nav_idx]
            to_ask_idices = [i for i, a in enumerate(answer_list) if a is not None]
            
            if len(to_ask_idices) > 0:
                # print("answer_list", answer_list)
                # print(nav_idx, "update instructions for ", to_ask_idices)
                dialog_inputs = self._language_variable_add_answer(obs, answer_list)
                dialog_txt_embeds = self.vln_bert('language', dialog_inputs)

                # Update dialog information
                language_inputs, txt_embeds = self.update_dialog(
                    language_inputs, txt_embeds, dialog_inputs, dialog_txt_embeds, to_ask_idices
                )

            # print(nav_idx, "language_inputs decoded", [self.tok.decode(language_inputs['txt_ids'][i]) for i in range(len(language_inputs['txt_ids']))])


            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    steps[i] += 1

            nav_inputs, pano_inputs = self._process_navigation_step(obs, gmaps, ended, nav_idx, language_inputs, txt_embeds)
            nav_probs, nav_vpids, nav_logits, nav_outs = self._nav_probs(nav_inputs)
            self._update_stop_scores(nav_probs, gmaps, obs, ended)
            next_viewpoints, ended = self._get_next_action_with_ref_path(obs, nav_historys, nav_idx, ended)
            self.make_equiv_action(next_viewpoints, gmaps, obs, traj_history)
            obs = self.env._get_obs()
        
        traj = [{
                'instr_id': ob['instr_id'],
                'path': [[ob['viewpoint']]],
                'details': {},
                'traj_history': traj_history[i]['path'],
            } for i, ob in enumerate(obs)]
        
        # print("history length", [len(traj_history[i]['path']) for i in range(len(traj_history))])
        return {
            'gmaps': gmaps,
            'traj': traj,
            'obs': obs,
            'steps': steps,
        }
    
    def set_navigation_history(self, set_navigation_history=False, use_dialog_history=False):
        if set_navigation_history:
            if use_dialog_history:
                ret =  self.init_graph_with_full_dialog(self.env._get_obs())
            else:
                ret =  self.init_graph_with_path_trajectory(self.env._get_obs())
        else:
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)

            gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
            traj = [{
                'instr_id': ob['instr_id'],
                'path': [[ob['viewpoint']]],
                'details': {},
                'wta_label': [],
                'wta_predict': [],
            } for ob in obs]

            ret = {
                'gmaps': gmaps,
                'traj': traj,
                'obs': obs,
                'steps': [0] * len(obs),
            }

        instr_encoding_list = [ob['instr_encoding'] for ob in ret['obs']]
        language_inputs = self._language_variable(instr_encoding_list)
        txt_embeds = self.vln_bert('language', language_inputs)
        
        ret['language_inputs'] = language_inputs
        ret['txt_embeds'] = txt_embeds
        ret['batch_size'] = len(ret['obs'])

        return ret

    def _update_graph_structure(self, obs, gmaps, ended):
        self._update_scanvp_cands(obs)
        for i, ob in enumerate(obs):
            if not ended[i]:
                gmaps[i].update_graph(ob)

    def rollout(self, train_ml=None, train_rl=False, reset=True, train_wta=False):
        # print("rollout", train_ml, self.feedback)

        torch.cuda.empty_cache()
        if reset:  # Reset env
            obs = self.env.reset()
        
        context = self.set_navigation_history(set_navigation_history=self.args.set_navigation_history, use_dialog_history=self.args.use_dialog_history)
        language_inputs = context['language_inputs']
        txt_embeds = context['txt_embeds']
        gmaps = context['gmaps']
        traj = context['traj']
        batch_size = context['batch_size']
        obs = context['obs']
        steps = context['steps']

        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)
        batch_size = len(obs)

        self._update_graph_structure(obs, gmaps, ended)


        ml_loss = 0.
        wta_loss = 0.

        for nav_idx in range(self.args.max_action_len):
            
            nav_inputs, pano_inputs = self._process_navigation_step(obs, gmaps, ended, nav_idx, language_inputs, txt_embeds, steps)
            nav_probs, nav_vpids, nav_logits, nav_outs = self._nav_probs(nav_inputs)


            nav_targets = None
            if train_ml is not None:
                # Supervised training
                nav_targets = self._teacher_action_r4r(
                    obs, nav_vpids, ended, 
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                    imitation_learning=(self.feedback=='teacher'), t=nav_idx, traj=traj
                )


                loss = self.criterion(nav_logits, nav_targets)
                # Check if loss is finite (not inf or nan)
                if torch.isfinite(loss):
                    ml_loss += loss
                else:
                    print(f"Warning: Loss is {loss}, ignoring this step. nav_idx: {nav_idx}")
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