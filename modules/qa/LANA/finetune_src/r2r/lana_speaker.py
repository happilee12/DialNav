from .agent_cmt_lana import Seq2SeqCMTAgent
import torch
import numpy as np
import networkx as nx
# from .data_utils import length2mask
# from utils.misc import length2mask
class LanaSpeaker(Seq2SeqCMTAgent):
    def __init__(self, args, env, tok, rank=0):
        super().__init__(args, env, tok, rank)

            
    def make_equiv_action_for_speaker(self, env, a_t, obs, max_hist_len, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """

        def take_action(i, name):
            if type(name) is int:  # Go to the next view
                env.env.sims[i].makeAction([name], [0], [0])
            else:  # Adjust
                env.env.sims[i].makeAction(*self.env_actions[name])
            
        def adjust(i, src_point, trg_point):
            src_level = src_point // 12  # The point idx started from 0
            trg_level = trg_point // 12
            while src_level < trg_level:  # Tune up
                take_action(i, 'up')
                src_level += 1
            while src_level > trg_level:  # Tune down
                take_action(i, 'down')
                src_level -= 1
            while env.env.sims[i].getState()[0].viewIndex != trg_point:  # Turn right until the target
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
                    state = env.env.sims[i].getState()[0]
                    for idx, loc in enumerate(state.navigableLocations):
                        if loc.viewpointId == select_candidate['viewpointId']:
                            take_action(i, idx)
                            state = env.env.sims[i].getState()[0]
                            if traj is not None:
                                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
                else:
                    action = action - 1  # 1 for global history token
                    target_vp = self.seq_vp_list[i][action]
                    current_vp = ob['viewpoint']
                    src_point = ob['viewIndex']
                    trg_point = self.seq_view_idx_list[i][action]
                    path = nx.single_source_shortest_path(self.graphs[i], current_vp)[target_vp]
                    state = env.env.sims[i].getState()[0]
                    for j in range(len(path) - 1):
                        # from path[j] to path[j+1]
                        for idx, loc in enumerate(state.navigableLocations):
                            if loc.viewpointId == path[j+1]:
                                take_action(i, idx)
                                state = env.env.sims[i].getState()[0]
                                if traj is not None:
                                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
                                break
                        else:
                            raise ValueError('no navigable location')
                    adjust(i, src_point, trg_point) 
    
    def _get_equiv_action(self, obs, next_viewpoints, ended, max_hist_len):
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == next_viewpoints[i]:
                        a[i] = k + max_hist_len
                        break
                else:  # Stop here
                    assert next_viewpoints[i] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate']) + max_hist_len
        return torch.from_numpy(a).cuda() 

    def get_history_and_actions_for_speaker(self, env, paths):
        add_eot = self.args.eot_token
        action_cache=[]
        obs = env._get_obs(t=0)
        ### given seq of viewpoints
        # target_instr = [ob['instr_id']  for ob in obs]
        batch_size = len(obs)
        # hist_lens = [1 for _ in range(batch_size)]                                                  # [b]
        hist_lens = torch.ones(batch_size, dtype=torch.long).cuda()  # [b]
        # action_lens = [0 for _ in range(batch_size)]
        action_lens = torch.zeros(batch_size, dtype=torch.long).cuda()  # [b]


        hist_pano_img = []
        hist_pano_ang = []
        hist_img = []
        prev_act = []

        traj = [{
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]
        ended = np.array([False] * batch_size)

        seen_paths = [[] for _ in range(batch_size)]

        for t in range(max(len(path) for path in paths)):  
            ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens, ob_pos = self._cand_pano_feature_variable(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    seen_paths[i].append(ob['viewpoint'])
            # ob_masks = length2mask(ob_lens).logical_not()

            # target = self._teacher_action(obs, ended, 0)
            # a_t = target
            next_viewpoints = [path[t+1] if t+1 < len(path) else path[-1] for path in paths]
            a_t = self._get_equiv_action(obs, next_viewpoints, ended, 0)


            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (ob_cand_lens[i]-1) or next_id == self.args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # if ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len-1)):
            # DDP error: RuntimeError: Expected to mark a variable ready only once.
            # It seems that every output from DDP should be used in order to perform correctly
            hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
            prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
            for i, next_id in enumerate(cpu_a_t):
                if next_id != -1:
                    prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
            prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

            hist_pano_img.append(hist_pano_img_feats) # [b,d] => [t, b, d]
            hist_pano_ang.append(hist_pano_ang_feats) # [b,d]
            hist_img.append(hist_img_feats)
            prev_act.append(prev_act_angle)

            for i, i_ended in enumerate(ended):
                if not i_ended:
                    hist_lens[i] += 1
                    action_lens[i] += 1
            # print(" obs .. ", [ob['viewpoint'] for ob in obs])

            # print("teacher path .. ", [ob['teacher_path'] for ob in obs])
            # print("cpu_a_t", cpu_a_t)

            self.make_equiv_action_for_speaker(env, cpu_a_t, obs, 0, traj)
            obs = env._get_obs(t=t+1)
            # print("updated obs .. ", [ob['viewpoint'] for ob in obs])
            
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break
            
        hist_pano_img_tensor = torch.stack(hist_pano_img, dim=0)
        hist_pano_img_tensor = hist_pano_img_tensor.permute(1, 0, 2, 3)  # ( batch_size, time_steps, feature_dim)
        hist_pano_ang_tensor = torch.stack(hist_pano_ang, dim=0).permute(1, 0, 2, 3)  # ( batch_size, time_steps, feature_dim)
        hist_img_tensor = torch.stack(hist_img, dim=0).permute(1, 0, 2)  # ( batch_size, time_steps, feature_dim)
        prev_act_tensor = torch.stack(prev_act, dim=0).permute(1, 0, 2)  # ( batch_size, time_steps, feature_dim)
        for idx, ob in enumerate(obs):
            # instr_id = ob['instr_id']  
            action_cache.append({
                'hist_pano_img_tensor': hist_pano_img_tensor[idx], # ( time_steps, feature_dim)
                'hist_pano_ang_tensor': hist_pano_ang_tensor[idx],
                'hist_img_tensor': hist_img_tensor[idx],
                'prev_act_tensor': prev_act_tensor[idx],
                'hist_len' : hist_lens[idx], 
                'action_len' : action_lens[idx] 
            })
        
        if add_eot:
            print("add eot token .. ", ended)
            for idx, ob in enumerate(obs):
                if ended[idx]: 
                    ## add eot token
                    hist_pano_img.append(torch.zeros_like(hist_pano_img_feats).cuda())
                    hist_pano_ang.append(torch.zeros_like(hist_pano_ang_feats).cuda())
                    hist_img.append(torch.zeros_like(hist_img_feats).cuda())
                    prev_act.append(torch.zeros_like(prev_act_angle).cuda())
                    hist_lens[idx] += 1
                    action_lens[idx] += 1
        
        hist_pano_img_tensor = torch.stack([action_cache[idx]['hist_pano_img_tensor'] for idx, ob in enumerate(obs)], dim=0)  # ( batch_size, time_steps, feature_dim) 
        hist_pano_ang_tensor = torch.stack([action_cache[idx]['hist_pano_ang_tensor'] for idx, ob in enumerate(obs)], dim=0)  # ( batch_size, time_steps, feature_dim) 
        hist_img_tensor = torch.stack([action_cache[idx]['hist_img_tensor'] for idx, ob in enumerate(obs)], dim=0)  # ( batch_size, time_steps, feature_dim) 
        prev_act_tensor = torch.stack([action_cache[idx]['prev_act_tensor'] for idx, ob in enumerate(obs)], dim=0)  # ( batch_size, time_steps, feature_dim) 
        hist_len = torch.tensor([action_cache[idx]['hist_len'] for idx, ob in enumerate(obs)]).cuda() 
        action_len = torch.tensor([action_cache[idx]['action_len'] for idx, ob in enumerate(obs)]).cuda() 
        hist_pano_img_tensor = hist_pano_img_tensor.permute(1, 0, 2, 3)  # ( time_steps, batch_size,  feature_dim)
        hist_pano_ang_tensor = hist_pano_ang_tensor.permute(1, 0, 2, 3)  # ( time_steps, batch_size,  feature_dim)
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

        return t_hist_inputs_list, t_action_inputs_list, hist_len, action_len, seen_paths 
  
  
    def say(self, scanIds, paths, env=None, max_given_length=1, max_generate_length=199, given=[]):
        GIVEN_LENGTH = max_given_length # self.args.max_given_len 
        DECODE_LENGTH = max_generate_length #self.args.max_instr_len 

        if env==None:
            raise Exception("env is required for LANA speaker")

        start_viewpoints = [path[0] for path in paths]
        start_headings = [3.14]*len(scanIds)
        env.reset(scanIds, start_viewpoints, start_headings, [])
        obs = env._get_obs()
        batch_size = len(obs)

        t_hist_inputs_list, t_action_inputs_list, hist_lens, action_lens, seen_paths  = self.get_history_and_actions_for_speaker(env, paths)


        hist_embeds = [self.vln_bert('history').expand(batch_size, -1, -1)] 
        action_embeds = []
        for t, action_input in enumerate(t_action_inputs_list):
            t_hist_inputs = t_hist_inputs_list[t]
            t_action_inputs =  t_action_inputs_list[t]
            t_hist_embeds = self.vln_bert(**t_hist_inputs)
            hist_embeds.append(t_hist_embeds)
            t_action_embeds = self.vln_bert(**t_action_inputs)
            action_embeds.append(t_action_embeds.unsqueeze(1))
 
        bs = len(obs)
        ended = torch.zeros(bs, dtype=torch.bool).cuda()

        words = torch.ones(bs, 1, dtype=torch.long) * self.cls_token_id     # [b,1]
        words = words.cuda()

        # ### Encode given text
        # if len(given) > 0:
        #     given_encoded_list = []
        #     given_token = 'Target : ' 
        #     start_token = 'Question : '
        #     if caption_type == 'answer':
        #         given_token = 'Question : ' 
        #         start_token = 'Answer : '
        #     for text in given:
        #         given_text = f"{given_token}{text} {start_token}"
        #         # print("given_text", given_text)

        #         if self.args.use_clip16:
        #             clip_pad_token_id = 0
        #             encoded = self.tokenizer.encode(given_text)[:GIVEN_LENGTH]
        #             padding_length = GIVEN_LENGTH - len(encoded)
        #             encoded = [clip_pad_token_id] * padding_length + encoded
        #             given_encoded_list.append(encoded)

        #     given_lengths = [len(given_encoded) for given_encoded in given_encoded_list]
        #     max_given_lengths = max(given_lengths)
        #     given_tensor = np.zeros((len(given_encoded_list), max_given_lengths))
        #     for i, given_item in enumerate(given_encoded_list):
        #         given_tensor[i, :GIVEN_LENGTH] = given_item
        #     given_tensor = torch.from_numpy(given_tensor)
        #     words = given_tensor.long().cuda()

        for i in range(DECODE_LENGTH):
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
        

        created_tokens_list = words.cpu().numpy() 
        # if len(given) > 0:
        #     created_tokens_list = created_tokens_list[:, GIVEN_LENGTH:]

        natural_language_output = []
        for inst in created_tokens_list: 
            if self.args.use_clip16:
                shrinked = self.tokenizer.shrink(inst)  # Shrink the words
                sentence = self.tokenizer.decode_sentence(shrinked)
                natural_language_output.append(sentence)
        return created_tokens_list, natural_language_output, seen_paths
