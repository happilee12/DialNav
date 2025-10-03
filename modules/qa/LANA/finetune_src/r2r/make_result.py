from datetime import datetime
import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
# from tensorboardX import SummaryWriter
import datetime
import sys
import wandb
import ast

sys.path.append(os.getcwd())

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from lana_models.vlnbert_init_lana import get_tokenizer
from r2r.agent_cmt_lana import Seq2SeqCMTAgent
from r2r.data_utils import ImageFeaturesDB, construct_instrs
from r2r.env import R2RBatch, R2RBackBatch
from r2r.parser import parse_args
from r2r.cap_eval.COCOEvalCap import COCOEvalCap


# def build_speaker_env(args, scan_list):
#     feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, use_clip16=args.use_clip16)
#     return SpeakerBatch(feat_db, args.connectivity_dir, scan_list)

def build_dataset(args, tok, rank=0, train=True, is_test=False):
    print("build dataset args", args)
    # tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, use_clip16=args.use_clip16)

    dataset_class = R2RBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    train_env = None
    aug_env = None
    if train:   
        train_instr_data = construct_instrs(
            f"{args.anno_dir}/train_inst.json", tokenizer=tok,max_given_len=args.max_given_len, max_instr_len=args.max_instr_len, max_action_len=args.max_action_len,
              use_clip16=args.use_clip16, caption_type=args.caption_type, target_path=args.caption_target_path, debug=args.debug
        )
        train_env = dataset_class(
            feat_db, train_instr_data, args.connectivity_dir, batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
            sel_data_idxs=None, name='train', anno_dir=args.anno_dir
        )
        if args.aug:
            aug_instr_data = construct_instrs(
                f"{args.anno_dir}/aug_train_inst.json", tokenizer=tok, max_given_len=args.max_given_len, max_instr_len=args.max_instr_len, max_action_len=args.max_action_len,
                        use_clip16=args.use_clip16,  caption_type=args.caption_type, target_path=args.caption_target_path, debug=args.debug
            )
            aug_env = dataset_class(
                feat_db, aug_instr_data, args.connectivity_dir, batch_size=args.batch_size,
                angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
                sel_data_idxs=None, name='aug', anno_dir=args.anno_dir
            )

    val_envs = {}
    val_env_names = ['val_seen', 'val_unseen']
    if args.test:
        val_env_names.append('test')
    # val_env_names = ['val_seen', 'val_unseen', 'val_train_seen']
    for split in val_env_names:
        val_instr_data = construct_instrs(
            f"{args.anno_dir}/{split}.json", tokenizer=tok, max_given_len=args.max_given_len, max_instr_len=args.max_instr_len, max_action_len=args.max_action_len, 
            use_clip16=args.use_clip16,  caption_type=args.caption_type, target_path=args.caption_target_path, debug=args.debug
        )
        if args.validation_size > 0:
           val_instr_data  = val_instr_data[:args.validation_size] 
        
        # print(split, " length: ", len(val_instr_data))


        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            anno_dir=args.anno_dir
        )
        evaluator = COCOEvalCap(val_instr_data)
        # evaluator = COCOEvalCap([split], tok, val_instr_data, use_clip16=args.use_clip16)
        val_envs[split] = (val_env, evaluator, split)

    return train_env, val_envs, aug_env

def make_result(args, tok, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = Seq2SeqCMTAgent
    first_env_in_envs = list(val_envs.keys())[0]
    agent = agent_class(args, val_envs[first_env_in_envs], tok, rank=rank)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    if args.resume_file is not None:
        file_path = args.resume_file
        iter = agent.load(file_path)
        if default_gpu:
            write_to_record_file("Loaded the listener model at iter %d from %s" % (iter, file_path), record_file)

    for env_name, (env, evaluator, name) in val_envs.items():
        agent.env = env
        infered_speech_path = os.path.join(args.log_dir, env_name+'_infered_speech.txt')
        path2inst = agent.valid_speaker(infered_speech_path)
        evaluator.evaluate(path2inst)
        
        # Log metrics to wandb
        if args.wandb_log:
            metrics = {f"{env_name}/{k}": v for k,v in evaluator.eval.items()}
            wandb.log(metrics)
            
        eval_str = f"Evaluation Results for {env_name}:"
        for metric, value in evaluator.eval.items():
            eval_str += f"\n{metric:<8}: {value:.4f}"
        write_to_record_file(eval_str + "\n", record_file)

        # if args.caption_type == 'answer':
        #     make_result_file(args, env_name)
        make_result_file(args, env_name, caption_type=args.caption_type)

def make_result_file(args, env_name, caption_type='answer'):
    ### read each line in os.path.join(args.log_dir, env_name+'_infered_speech.txt')
    def read_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    org_annotations = read_json(os.path.join(args.anno_dir, f'{env_name}.json'))
    org_annotations = {item['instr_id']: item for item in org_annotations}

    target_key = 'a' if caption_type == 'answer' else 'q'

    with open(os.path.join(args.log_dir, env_name+'_infered_speech.txt'), 'r') as f:
        for line in f:
            parsed = ast.literal_eval(line)
            path_id = parsed['path_id']
            created = parsed['created']
            org_item = org_annotations[path_id]
            org_item['org_'+target_key] = org_item[target_key]
            org_item[target_key] = created

    org_annotations = list(org_annotations.values())
    replaced_all = ['org_a' in item for item in org_annotations]
    print(f"replaced_all: {sum(replaced_all)} / {len(org_annotations)}")

    os.makedirs(os.path.join(args.log_dir, 'lana_instruction'), exist_ok=True)
    with open(os.path.join(args.log_dir, 'lana_instruction', env_name+'.json'), 'w') as f:
        json.dump(org_annotations, f, indent=4)


def main():
    args = parse_args()

    if args.debug:
        args.batch_size = 2
        args.target_batch_size =2
        args.wandb_log = False
        args.log_every = 10
        args.iters = 20

    if args.wandb_log:
        wandb.init(
            project=args.wandb_project,
            id=args.id,
            config=args
        )
    tok = get_tokenizer(args)

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    _, val_envs, _ = build_dataset(args, tok, rank=rank, train=False)

    make_result(args, tok, val_envs, rank=rank)


if __name__ == '__main__':
    main()
