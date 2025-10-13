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
import shutil

# def build_speaker_env(args, scan_list):
#     feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, use_clip16=args.use_clip16)
#     return SpeakerBatch(feat_db, args.connectivity_dir, scan_list)

def build_dataset(args, tok, rank=0, is_test=False):
    print("build dataset args", args)
    # tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, use_clip16=args.use_clip16)

    dataset_class = R2RBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
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
            f"{args.aug_data_dir}/aug_train_inst.jsonl", tokenizer=tok, max_given_len=args.max_given_len, max_instr_len=args.max_instr_len, max_action_len=args.max_action_len,
            # f"{args.aug_data_dir}/aug_train_inst.json", tokenizer=tok, max_given_len=args.max_given_len, max_instr_len=args.max_instr_len, max_action_len=args.max_action_len,
                      use_clip16=args.use_clip16,  caption_type=args.caption_type, target_path=args.caption_target_path, debug=args.debug
        )
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
            sel_data_idxs=None, name='aug', anno_dir=args.anno_dir
        )
    else:
        aug_env = None

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


def train(args, tok, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        # writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = Seq2SeqCMTAgent
    listener = agent_class(args, train_env, tok, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listener.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter),
                record_file
            )

    # first evaluation
    print(datetime.datetime.now())

    best_eval = {
        'val_seen_Bleu_4': 0, 
        'val_unseen_Bleu_4': 0, 
        'val_avg_Bleu_4': 0,  
        # 'val_unseen_Bleu_1': 0, 
        # 'val_unseen_CIDEr': 0, 
        # 'val_unseen_ROUGE_L': 0 
    }
    if args.eval_first:
        print('test language metric')
        eval = {"iter": start_iter, "step": start_iter}
        for env_name, (env, evaluator, name) in val_envs.items():
            listener.env = env
            infered_speech_path = os.path.join(args.log_dir, env_name+'_initial_speech.txt')
            path2inst = listener.valid_speaker(infered_speech_path)
            evaluator.evaluate(path2inst)
            eval_str = evaluator.eval.items()
            metrics= ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'CIDEr' ]
            
            for metric in metrics:
                eval["%s_%s"%(env_name, metric)] = evaluator.eval[metric]
            write_to_record_file(str(eval_str), record_file)
            if args.wandb_log:
                wandb.log(eval)
    print(datetime.datetime.now())

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )


    accumulation_steps = 1
    if args.target_batch_size > args.batch_size:
        accumulation_steps = args.target_batch_size // args.batch_size
        print(f"batch size: {args.batch_size}, target batch size: {args.target_batch_size} -> accumulation_steps: {accumulation_steps}")

    for idx in range(start_iter, start_iter + args.iters, args.log_every):
        listener.logs = defaultdict(list)
        interval = min(args.log_every, start_iter +args.iters - idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            task = args.task_name
            contrastive_loss = 0.0
            caption_loss = 0.0
            if task == "caption":
                # jdx_length = len(range(interval // 2))
                interveral = range(interval)
                for jdx in interveral:
                    listener.env = train_env
                    caption_loss = listener.train_speaker(accumulation_steps, args.use_cache)  
                    # print("caption_loss", caption_loss)
                    if default_gpu:
                        print_progress(jdx, len(interveral), prefix='Progress:', suffix='Complete', bar_length=50)
        else:
            contrastive_loss = 0.0
            caption_loss = 0.0
            jdx_list = range(0, interval, (1+args.aug_times))
            total_steps = len(jdx_list)
            for jdx_idx, jdx in enumerate(jdx_list):
                task = 'caption'
                # if args.mix_task:
                #     weights = torch.Tensor([args.caption_task_weight, args.vln_task_weight])
                #     task_id = torch.multinomial(weights, 1, replacement=True)
                #     if task_id == 0:
                #         task = 'caption'
                #     elif task_id == 1:
                #         task = 'regular'
                    
                if task == "caption":
                    listener.env = train_env
                    train_loss = listener.train_speaker(accumulation_steps, args.use_cache)
                    caption_loss += train_loss
                    if args.debug:
                        print(f"training with train {jdx} / . Loss: {train_loss}")
                    listener.env = aug_env
                    for aug_idx in range(args.aug_times):
                        aug_loss = listener.train_speaker(accumulation_steps, args.aug_use_cache)
                        if args.debug:
                            print(f"training with aug {jdx} / aug_idx: {aug_idx}. Loss: {aug_loss}")
                        caption_loss += aug_loss
                    caption_loss /= (1+args.aug_times)
                # elif task == "contrastive":
                #     listener.env = train_env
                #     contrastive_loss1 = listener.train_cont(1)
                #     listener.env = aug_env
                #     contrastive_loss2 = listener.train_cont(1)
                #     contrastive_loss = (contrastive_loss1 + contrastive_loss2) / 2.0
                # else:
                #     # Train with GT data
                #     listener.env = train_env
                #     listener.train(1, feedback=args.feedback)
                #     # Train with Augmented data
                #     listener.env = aug_env
                #     listener.train(1, feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx_idx+1, total_steps, prefix='Progress:', suffix='Complete', bar_length=50)

        # Run validation
        if args.task_name == "caption":
            eval = {
                "caption_loss": caption_loss,
                "iter": iter,
                "step": iter
                # "data_size": iter*args.batch_size
            }
            for env_name, (env, evaluator, name) in val_envs.items():
                write_to_record_file('test language metric '+env_name, record_file)
                # if name == "val_train_seen":
                #     continue
                listener.env = env
                infered_speech_path = os.path.join(args.log_dir, env_name+'_'+str(iter)+'_speech.txt')
                path2inst = listener.valid_speaker(infered_speech_path)
                evaluator.evaluate(path2inst)
                eval_str = evaluator.eval.items()
                write_to_record_file(str(eval_str), record_file)
                listener.env = env
                # write_to_record_file(str(eval_str), record_file)

                ''' save model weight'''
                if default_gpu:
                    write_to_record_file(
                                '\nEvaluatiing result ...  %s %s' % (env_name, str(iter)), record_file
                            )
                    # listener.save(idx, os.path.join(args.ckpt_dir, "ckpt_%s"%(str(iter)))) 
                    # metrics= ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'CIDEr']
                    metrics= ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'CIDEr' ]
                    for key in metrics:
                        eval["%s_%s"%(env_name, key)] = evaluator.eval[key] 
                        if "%s_%s"%(env_name, key) in best_eval and  evaluator.eval[key] > best_eval["%s_%s"%(env_name, key)]:
                            write_to_record_file(
                                '\n Better here! %s %s %s' % (env_name, str(key), str(evaluator.eval[key])), record_file
                            )
                            best_eval["%s_%s"%(env_name, key)] =  evaluator.eval[key]
                            if env_name == 'val_unseen' and not args.debug:
                                listener.save(iter, os.path.join(args.ckpt_dir, "best_%s_%s"%(env_name, key)))
                                # listener.save(idx, os.path.join(args.ckpt_dir, "best_%s_%s_%s"%(env_name, key, str(evaluator.eval[key]))))
                                write_to_record_file(
                                    '\nSaved to %s' % args.ckpt_dir, record_file
                                )
                    listener.save(idx, os.path.join(args.ckpt_dir, "latest"))
                    
                    if 'val_seen_Bleu_4' in eval and 'val_unseen_Bleu_4' in eval:
                        current_avg = (eval['val_seen_Bleu_4'] + eval['val_unseen_Bleu_4']) / 2.0
                        eval['val_avg_Bleu_4'] = current_avg
                        
                        if current_avg > best_eval['val_avg_Bleu_4']:
                            write_to_record_file(
                                '\n Better average! val_avg_Bleu_4: %s (prev: %s)' % (str(current_avg), str(best_eval['val_avg_Bleu_4'])), record_file
                            )
                            best_eval['val_avg_Bleu_4'] = current_avg
                            if not args.debug:
                                listener.save(iter, os.path.join(args.ckpt_dir, "best_val_avg_Bleu_4"))
                                write_to_record_file(
                                    '\nSaved best average model to %s' % args.ckpt_dir, record_file
                                )
                    
                    # if aug_env is not None and iter % (args.log_every * 5) == 0:
                    #     listener.save(iter, os.path.join(args.ckpt_dir, f"cp_{iter}"))
                    if (iter - start_iter) % (args.log_every * 20) == 0:
                        listener.save(iter, os.path.join(args.ckpt_dir, f"cp_{iter}"))
                    # if not args.debug:      
                    #     listener.save(iter, os.path.join(args.ckpt_dir, "latest"))

            print("log to wandb", args.wandb_log, iter, eval)
            if args.wandb_log:
                try:
                    wandb.log(eval)
                    print(f"Successfully logged to wandb at iter {iter}")
                except Exception as e:
                    print(f"Failed to log to wandb at iter {iter}: {e}", eval)

def valid(args, tok, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = Seq2SeqCMTAgent
    agent = agent_class(args, train_env, tok, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

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

        if args.caption_type == 'answer':
            make_file_for_navigation_meric(args, env_name)

def make_file_for_navigation_meric(args, env_name):
    ### read each line in os.path.join(args.log_dir, env_name+'_infered_speech.txt')
    def read_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    org_annotations = read_json(os.path.join(args.anno_dir, f'{env_name}.json'))
    org_annotations = {item['instr_id']: item for item in org_annotations}

    with open(os.path.join(args.log_dir, env_name+'_infered_speech.txt'), 'r') as f:
        for line in f:
            parsed = ast.literal_eval(line)
            path_id = parsed['path_id']
            created = parsed['created']
            org_item = org_annotations[path_id]
            org_item['org_a'] = org_item['a']
            org_item['a'] = created

    org_annotations = list(org_annotations.values())
    replaced_all = ['org_a' in item for item in org_annotations]
    print(f"replaced_all: {sum(replaced_all)} / {len(org_annotations)}")

    os.makedirs(os.path.join(args.log_dir, 'nav_instruction'), exist_ok=True)
    with open(os.path.join(args.log_dir, 'nav_instruction', env_name+'.json'), 'w') as f:
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
            group='lana',
            id=args.id,
            config=args,
            resume=True
        )
    tok = get_tokenizer(args)

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env = build_dataset(args, tok, rank=rank)

    if not args.test:
        train(args, tok, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        valid(args, tok, train_env, val_envs, rank=rank)


if __name__ == '__main__':
    main()
