from parser import parse_args
import wandb
import torch
from holistic_utils.distributed import init_distributed, is_default_gpu
from holistic_utils.misc import set_random_seed
from holistic_utils.data_utils import construct_instrs
from holistic_models.ScaleVLN.ScaleVLN import ScaleVLNModel
from ModularNavigator import ModularNavigator
from ModularGuide import ModularGuide
from holistic_models.FixedInterval import FixedIntervalWtaModule
from holistic_models.ConfidenceThresholding import ConfidenceThresholdingWtaModule
from holistic_models.FixedResponse import FixedAnswerGeneration, FixedQuestionGeneration
from holistic_models.LANA.LANA import LANA
from holistic_models.GCNLoc.GCNLoc import GCNLocModel
import time
import numpy as np
from evaluator import Evaluator
import copy
import os
import json
from transformers import logging
import sys
logging.set_verbosity_error()

def get_tokenizer():
    from transformers import AutoTokenizer
    cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def load_instruction_data(args, target_envs, tokenizer):
    env_instructions = {}
    for split in target_envs:
        if split == "val_seen":
            annotation_paths = args.val_seen_anno_paths
        elif split == "val_unseen":
            annotation_paths = args.val_unseen_anno_paths
        elif split == "test":
            annotation_paths = args.test_anno_paths
        else:
            raise ValueError(f"Invalid split: {split}")
        
        annotation_paths = annotation_paths.split(",")
        instruction_data = construct_instrs(annotation_paths, tokenizer, args.max_instr_len)
        print(f"Loaded instruction data for split: {split} ({annotation_paths}) with length: {len(instruction_data)}")
        env_instructions[split] = instruction_data
    return env_instructions


def dialNav(navigator, 
            guide,
            mode,
            max_action_len=50):
    navigator.set_next_batch()
    obs = navigator.get_obs()
    batch_size = len(obs)


    traj = [{
            'scan': ob['scan'],
            'start_pano': ob['viewpoint'],
            'gt_path': ob['gt_path'],
            'end_panos': ob['end_panos'],
            'target': ob['instruction'],
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'dialog': [],
            'navigation_detail': []
        } for idx, ob in enumerate(obs)]
    navigator.initialize_nav(obs)

    ask = np.array([False] * batch_size)


    question_seen_path = None
    answer_seen_path = None
    for step in range(max_action_len):
        next_vp_ids, ended, nav_probs, instrucion_for_this_nav, nav_outs = navigator.get_next_action(step, obs)
        next_vp_ids_before_dialog = copy.deepcopy(next_vp_ids)
        ended_before_this_step = copy.deepcopy(ended)

        ## details
        nav_probs_cache = nav_probs.clone()
        c = torch.distributions.Categorical(nav_probs)
        c_cache = torch.distributions.Categorical(nav_probs_cache)

        if mode == 'navonly':
            ask = np.array([False] * batch_size)
        else:
            ask = navigator.wta(step, nav_probs, nav_outs)
            
        to_ask_indices = [index for index, value in enumerate(ask) if value and not ended[index]]
        need_dialog = len(to_ask_indices) > 0
        if need_dialog:
            print(f"[Step {step}] to_ask_indices: {to_ask_indices}")
        scanIds = [obs[i]['scan'] for i in range(batch_size)]
        viewpoints = [obs[i]['viewpoint'] for i in range(batch_size)]
        # goals = [obs[i]['end_panos'] for i in range(batch_size)]
        goals = [obs[i]['gt_path'][-1] for i in range(batch_size)]

        if need_dialog:
            if mode == 'gt_loc':
                questions = ["Where should I go?" for i in range(batch_size)]
                localized_viewpoints = [obs[i]['viewpoint'] for i in range(batch_size)]
            else:
                questions, question_seen_path = navigator.ask(scanIds, viewpoints)
                # print("ask1 questions", questions, question_seen_path)
                # questions, question_seen_path = navigator.ask2(scanIds, viewpoints, goals=goals)
                # print("ask 2 questions", questions, question_seen_path)
                localized_viewpoints = guide.localize(scanIds, questions)
                # print("localized viewpoints", viewpoints)
            
            paths = [guide._choose_path(scanId, viewpoint, [goal]) for scanId, viewpoint, goal in zip(scanIds, localized_viewpoints, goals)]

            answers, answer_seen_path = guide.answer(scanIds, localized_viewpoints, paths)
            navigator.update_instruction(to_ask_indices, questions, answers)

            # raise Exception("stop here")
            
            # update navigation actions with new dialog
            next_vp_ids, ended, nav_probs, instrucion_for_this_nav, nav_outs = navigator.get_next_action(step, obs)

        obs, paths = navigator.navigate(next_vp_ids, obs, ended, traj)
        just_ended = ended & ~ended_before_this_step

        ### update trajectory log
        c = torch.distributions.Categorical(nav_probs)
        for i in range(batch_size):
            if ended[i] and not just_ended[i]:
                continue
            
            navigation_detail_item = {
                'nav_idx': step,
                'ask': False,
                'instruction': instrucion_for_this_nav[i],
                'gt_viewpoint': viewpoints[i],
                'next_vp_ids': next_vp_ids[i],
                'ended': ended[i],
                # 'nav_probs': nav_probs[i],
                'entropy': c.entropy()[i].item(),
            }
            if i in to_ask_indices:
                navigation_detail_item['ask'] = True
                navigation_detail_item['question'] = questions[i]
                navigation_detail_item['localized_viewpoint'] = localized_viewpoints[i]
                navigation_detail_item['answer'] = answers[i]
                # navigation_detail_item['gt_viewpoint'] = viewpoints[i]
                if question_seen_path:
                    navigation_detail_item['question_seen_path'] = question_seen_path[i]
                if answer_seen_path:
                    navigation_detail_item['answer_seen_path'] = answer_seen_path[i]
                navigation_detail_item['vp_before_dialog'] = next_vp_ids_before_dialog[i]
                navigation_detail_item['entropy_before_dialog'] = c_cache.entropy()[i].item()
                navigation_detail_item['entropy_diff'] = navigation_detail_item['entropy'] - navigation_detail_item['entropy_before_dialog']
            traj[i]['navigation_detail'].append(navigation_detail_item)

            ## already processed in make_equiv_action
            # if not just_ended[i]:
            #     traj[i]['path'].append(paths[i])

        if all(ended):
            break

    return traj

def run(navigator, guide, max_action_len, mode, env_name, output_file):
    print("evaluating on env: ", env_name)
    start_time = time.time()

    # Set the target environment
    navigator.set_target_env(env_name)

    # Reset the data index to beginning of epoch. 
    navigator.reset_epoch()
    results = {}

    index = 1
    finished = False
    while not finished:
        print(f"Processing data {index}")
        index += 1
        # if index > 2:
        #     finished = True
        
        trajectories = dialNav(
            navigator, 
            guide, 
            max_action_len=max_action_len, 
            mode=mode,
        )
        for traj in trajectories:
            if traj['instr_id'] in results:
                finished = True
            if not finished:
                results[traj['instr_id']] = traj
            
        ### make output in list
        output = [{'instr_id': k, **v} for k, v in results.items()]
        ## save output to json
        with open(output_file, "w") as f:
            json.dump(output, f, default=lambda x: x.item() if isinstance(x, (bool, np.bool_)) else x)
    print("finished all trajectories", len(results))
    print("time taken: ", time.time() - start_time, "seconds")

    
    return output
            
def setWta(wta_mode, navigation_model=None):
    if wta_mode.startswith('every'):
        print("Setting wta to every interval")
        return FixedIntervalWtaModule(interval=int(wta_mode.split('_')[1]))
    elif wta_mode.startswith('ct'):
        return ConfidenceThresholdingWtaModule(threshold=float(wta_mode.split('_')[1]))
    elif wta_mode == 'navigation_model':
        return navigation_model
    
def setAgents(args, target_envs, env_instructions, evaluator, scans):

    if args.nav_model == 'ScaleVLN':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        modules_path = os.path.join(current_dir, '../../../modules/nav/ScaleVLN/map_nav_src')
        sys.path.insert(0, modules_path)
        ### Initialize Modules
        navigation_model_args = {
            'batch_size': args.batch_size, 
            'basepath': args.basepath, 
            'resume_file': args.nav_resume_file,
            'act_visited_nodes': args.nav_act_visited_nodes,
            'connectivity_dir': args.connectivity_dir,
            'wta_question_threshold': args.nav_wta_question_threshold,
        }
        navigation_model = ScaleVLNModel(args.basepath, navigation_model_args)
        navigation_model.eval()
        navigation_model.set_envs(target_envs, env_instructions)
    else:
        raise ValueError(f"Invalid navigation model: {args.nav_model}")
    localization_model = None
    question_model = None
    answer_model = None
    wta_model = None
    if args.mode != 'navonly':
        wta_model = setWta(args.wta_mode, navigation_model)
        # wta_model = FixedIntervalWtaModule(interval=32) 
        # wta_model = ConfidenceThresholdingWtaModule(threshold=0.5) 
        # question_model = FixedQuestionGeneration(question='')
        # answer_model = FixedAnswerGeneration(response='Go straight')
        answer_model = LANA(args.basepath, {
            'scan_list': scans,
            'resume_file': args.ag_resume_file,
            'eot_token': args.ag_eot_token,
            'connectivity_dir': args.connectivity_dir,
            'bpe_path': args.qa_clip_tokenizer_path,
        }, type='ag') 

        if args.mode != 'gt_loc':
            # question_model = FixedQuestionGeneration(question='')
            question_model = LANA(args.basepath, {
                'scan_list': scans,
                'resume_file': args.qg_resume_file,
                'connectivity_dir': args.connectivity_dir,
                'bpe_path': args.qa_clip_tokenizer_path,
            }, type='qg')

            if args.loc_model == 'GCN':
                localization_model = GCNLocModel(args.basepath, {
                    'eval_ckpt': args.loc_resume_file,
                    'panofeat_dir': args.loc_node_feats_dir,
                    'geodistance_file': args.loc_geodistance_nodes_path,
                    'connect_dir': args.connectivity_dir+"/",
                    'embedding_dir': args.loc_embedding_dir,
                    'bert_enc': args.loc_bert_enc,
                })
            else:
                raise ValueError(f"Invalid localization model: {args.loc_model}")

    env_infos = {"shortest_distances": evaluator.shortest_distances, "shortest_paths": evaluator.shortest_paths}
    guide_agent = ModularGuide(args, answer_model, localization_model, env_infos)
    navigator_agent = ModularNavigator(args, navigation_model, wta_model, question_model)
    return navigator_agent, guide_agent



def main():
    print("run parser")
    args = parse_args()
    target_envs = args.env_names.split(",")


    ### make output path
    os.makedirs(args.output_path, exist_ok=True)

    print("MAIN ARGS")
    print("args", args)
    args_log_file = f"{args.output_path}/args.txt"
    with open(args_log_file, "w") as f:
        f.write("--- MAIN ARGS --- \n")
        f.write("Time: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        for key, value in vars(args).items():
            if isinstance(value, (np.int64, np.float64)):
                value = value.item()
            f.write(f"{key}: {value}\n")
        f.write("\n\n")

    if args.wandb_log:
        wandb.init(
            project=args.wandb_project,
            config=args,
            id=args.id, 
        )
    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0


    set_random_seed(args.seed + rank)

    tokenizer = get_tokenizer()
    env_instructions = load_instruction_data(args, target_envs, tokenizer)
    # env_instructions['val_seen'] = env_instructions['val_seen'][:8]

    ## set up evaluator
    scans = list(set([item['scan'] for env_name in target_envs for item in env_instructions[env_name]]))
    evaluator = Evaluator(args.connectivity_dir, scans)

    navigator_agent, guide_agent = setAgents(args, target_envs, env_instructions, evaluator, scans)

    with open(args_log_file, "a") as f:
        targets = {
            "navigation": navigator_agent.navigation_model.args,
            "answer_generation": guide_agent.answer_model.args,
        }
        if args.mode == 'holistic':
            targets['question_generation'] = navigator_agent.question_generation_model.args
            targets['localization'] = guide_agent.localization_model.args
        
        f.write("--- NAVIGATOR ARGS --- \n")
        for key, value in targets.items():
            f.write(f"{key}: \n")
            for key, value in vars(value).items():
                if isinstance(value, (np.int64, np.float64)):
                    value = value.item()
                f.write(f"{key}: {value}\n")
            f.write("\n\n")

    for env_name in target_envs:
        metrics_acc = {}
        avg_metrics_acc = {}
        output = run(
            navigator_agent, 
            guide_agent, 
            max_action_len=args.max_action_len,
            mode=args.mode,
            env_name=env_name,
            output_file=f"{args.output_path}/{env_name}.json"
        )


        for item in output:
            item['nav_error'] = float(evaluator.get_shortest(item['scan'], item['path'][-1][-1], item['end_panos']))
            for detail in item['navigation_detail']:
                if 'localized_viewpoint' in detail:
                    detail['loc_error'] = float(evaluator.get_shortest(item['scan'], detail['gt_viewpoint'], [detail['localized_viewpoint']]))


        ## save output to json
        with open(f"{args.output_path}/{env_name}.json", "w") as f:
            json.dump(output, f, default=lambda x: x.item() if isinstance(x, (bool, np.bool_)) else x)

        avg_metrics, metrics = evaluator.eval_metrics(output)
        metrics_acc[env_name] = metrics
        avg_metrics_acc[env_name] = avg_metrics
        avg_metrics_acc[env_name]['Agg'] = f"{','.join([str(round(avg_metrics_acc[env_name][key], 2)) for key in ['sr','oracle_sr','spl','nav_error','steps','dtc','le']])}"

        with open(f"{args.output_path}/avg_metrics_{env_name}.json", "w") as f:
            json.dump({'avg_metrics_acc': avg_metrics_acc[env_name]}, f, default=lambda x: x.item() if isinstance(x, (np.int64, np.float64)) else x)
        with open(f"{args.output_path}/metrics_{env_name}.json", "w") as f:
            json.dump({'metrics_acc': metrics_acc[env_name]}, f, default=lambda x: x.item() if isinstance(x, (np.int64, np.float64)) else x)

        if args.wandb_log:
            metrics_dict = {}
            for env in avg_metrics_acc:
                for key, value in avg_metrics_acc[env].items():
                    if isinstance(value, float):
                        metrics_dict[f"{env}_{key}"] = round(value, 2)
                    else:
                        metrics_dict[f"{env}_{key}"] = value
            wandb.log(metrics_dict)

if __name__ == '__main__':
    main()