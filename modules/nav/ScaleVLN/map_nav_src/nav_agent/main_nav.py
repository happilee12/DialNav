import os
import json
import time
import numpy as np
from collections import defaultdict

import torch

import sys
sys.path.append("..")
sys.path.append(".")

from sv_utils.misc import set_random_seed
from sv_utils.logger import write_to_record_file, print_progress, timeSince
from sv_utils.distributed import init_distributed, is_default_gpu
from sv_utils.distributed import all_gather, merge_dist_results

from sv_utils.data import ImageFeaturesDB, ImageFeaturesDB2
from nav_agent.data_utils import construct_instrs
from nav_agent.env import NDHNavBatch
from nav_agent.parser import parse_args #아직 추가 안함

from sv_models.vlnbert_init import get_tokenizer
from nav_agent.agent import GMapNavAgent, RandomAgent, GMapNavAgnetWta
from nav_agent.context_agent import GMapWithContextAgent

import wandb

def build_dataset(args, rank=0, val_only=False, aug=False, except_train_seen=False, is_test=False, 
                  append_q=False, append_history=False, record_file=None):
    tok = get_tokenizer(args)
    val_feat_db = ImageFeaturesDB(args.val_ft_file, args.image_feat_size)
    if val_only and except_train_seen:
        train_feat_db = None
    else:
        train_feat_db = ImageFeaturesDB2([args.mp3d_ft_files], args.image_feat_size)
        print("loaded aug train_feat_db")

    dataset_class = NDHNavBatch
    val_env_names = ['val_seen', 'val_unseen', 'test']
    if not except_train_seen:
        val_env_names.append('val_train_seen')
 
    val_envs = {}
    for split in val_env_names:
        val_instr_data, metadata = construct_instrs(
            args.data_dir, args.dataset, [split], tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            append_q=append_q, append_history=append_history, maximum_navigation_history_length=args.maximum_navigation_history_length
        )
        write_to_record_file(split + '\n\n'+ json.dumps(metadata, default=(lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else None))+ '\n\n', record_file)

        val_env = dataset_class(
            val_feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            use_nav_turn_path=args.extend_wta,
        )
        val_envs[split] = val_env

    train_env = None
    if not val_only:
        train_instr_data, metadata = construct_instrs(
            args.data_dir, args.dataset, ['train_inst'], tokenizer=args.tokenizer, max_instr_len=args.max_instr_len, is_test=is_test,
            append_q=append_q, append_history=append_history, maximum_navigation_history_length=args.maximum_navigation_history_length
        )
        write_to_record_file('train \n\n'+ json.dumps(metadata, default=(lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else None)) + '\n\n', record_file)
        
        train_env = dataset_class(
            train_feat_db, train_instr_data, args.connectivity_dir,
            batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None, name='train',
            use_nav_turn_path=(args.extend_wta or args.use_nav_turn_path)
        )

    aug_train_env = None
    if aug:
        aug_train_instr_data, metadata = construct_instrs(
            args.aug_data_dir, args.dataset, ['aug_train_inst'], tokenizer=args.tokenizer, max_instr_len=args.max_instr_len, is_test=is_test,
            append_q=append_q, append_history=append_history, maximum_navigation_history_length=args.maximum_navigation_history_length
        )
        write_to_record_file('Aug Train \n\n'+ json.dumps(metadata, default=(lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else None)) + '\n\n', record_file)
        
        aug_train_env = dataset_class(
            train_feat_db, aug_train_instr_data, args.connectivity_dir,
            batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None, name='train',
            use_nav_turn_path=(args.extend_wta or args.use_nav_turn_path)
        )

    return train_env, val_envs, aug_train_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    if args.set_navigation_history:
        agent_class = GMapWithContextAgent
        print("setting agent_class to GMapWithContextAgent")
    elif args.extend_wta:
        agent_class = GMapNavAgnetWta
        print("setting agent_class to GMapNavAgnetWta")
    else:
        agent_class = GMapNavAgent
        print("setting agent_class to GMapNavAgent")

    print("start training ... ")
    listner = agent_class(args, train_env, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        wandb_log = { "iter": 0 }
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results(wta_test=args.extend_wta)
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds, wta_mode=args.extend_wta)
                print("score_summary (eval_first)", score_summary)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    wandb_log[env_name+"_"+metric] = val
                print("wandb_log", wandb_log)

        if default_gpu:
            write_to_record_file(loss_str, record_file)
            if args.wandb_log:
                wandb.log(wandb_log)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {
        'val_unseen': { "sr": 0, "state_sr":"", "wta_accuracy": 0, "state_wta_accuracy":""},
        'val_seen': { "sr": 0, "state_sr":"", "wta_accuracy": 0, "state_wta_accuracy":""},
        'val_avg': { "sr": 0, "state_sr":""} # Added for tracking average SR
    }
    data_size = 0
    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        print("idx", idx, start_iter, start_iter+args.iters, args.log_every)
        listner.logs = defaultdict(list)
        interval = min(args.log_every, start_iter+args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        # print every grad_acc, eval every interval*grad_acc
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback, grad_accum_steps=args.grad_accum_steps)
            data_size += args.batch_size*args.grad_accum_steps
        else:
            # print every (1+aug_times)*grad_acc, eval every interval*grad_acc/(1+aug_times)    
            jdx_length = len(range(interval // (args.aug_times+1)))
            for jdx in range(interval // (args.aug_times+1)):
                # Train with GT data
                listner.env = train_env
                listner.train(1, feedback=args.feedback, grad_accum_steps=args.grad_accum_steps)

                # Train with Augmented data
                # two aug one GT
                listner.env = aug_env
                listner.train(args.aug_times, feedback=args.feedback, grad_accum_steps=args.grad_accum_steps)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)
                
                data_size += args.batch_size*args.grad_accum_steps*(args.aug_times+1)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total

            if 'wta_loss' in listner.logs:
                wta_loss = sum(listner.logs['wta_loss']) / max(len(listner.logs['wta_loss']), 1)
                write_to_record_file("\nwta_loss %.4f" % (wta_loss), record_file)
            # Add OOM count to logging
            oom_log = ""
            if 'oom_count' in listner.logs:
                oom_count = sum(listner.logs['oom_count'])
                oom_log = f", oom_count {oom_count}"
            
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f%s" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss, oom_log),
                record_file
            )
            

        # Run validation
        loss_str = "iter {}".format(iter)
        wandb_log = {"loss": IL_loss, "length": length, "entropy": entropy, "iter": iter, "data_size": data_size}
        
        # Add OOM count to wandb logging
        if 'oom_count' in listner.logs:
            oom_count = sum(listner.logs['oom_count'])
            wandb_log['oom_count'] = oom_count

        ### WTA Loss log
        if 'wta_loss' in listner.logs:
            wandb_log['wta_loss'] = wta_loss

        current_sr_sum = 0
        env_count = 0

        for env_name, env in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            torch.cuda.empty_cache() # Clear cache after testing
            
            preds = listner.get_results(wta_test=args.extend_wta)
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds, wta_mode=args.extend_wta)
                print("score_summary", score_summary)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    wandb_log[env_name+"_"+metric] = val
                print("wandb_log", wandb_log)

                # select model by gp
                if env_name in best_val:
                    if args.extend_wta:
                        listner.save(idx, os.path.join(args.ckpt_dir, "iter_%d" % (idx)))
                        if 'wta_accuracy' in score_summary and score_summary['wta_accuracy'] >= best_val[env_name]['wta_accuracy']:
                            best_val[env_name]['wta_accuracy'] = score_summary['wta_accuracy']
                            listner.save(idx, os.path.join(args.ckpt_dir, "best_%s_wta" % (env_name)))
                    else:
                        if score_summary['sr'] >= best_val[env_name]['sr']:
                            best_val[env_name]['sr'] = score_summary['sr']
                            best_val[env_name]['state_sr'] = 'Iter %d %s' % (iter, loss_str)
                            listner.save(idx, os.path.join(args.ckpt_dir, "best_%s_sr" % (env_name)))
                        
                        # Calculate average SR
                        if env_name in ['val_seen', 'val_unseen']:
                            current_sr_sum += score_summary['sr']
                            env_count += 1

        # Save model for best average SR
        if default_gpu and env_count == 2:
            current_sr_avg = current_sr_sum / env_count
            if current_sr_avg >= best_val['val_avg']['sr']:
                best_val['val_avg']['sr'] = current_sr_avg
                best_val['val_avg']['state_sr'] = 'Iter %d %s' % (iter, loss_str)
                listner.save(idx, os.path.join(args.ckpt_dir, "best_avg_sr"))
        
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))
            if args.wandb_log:
                wandb.log(wandb_log)
            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state_sr'], record_file)


def valid(args,  val_envs, rank=-1, instr_iter=0):
    print("start valid .. ")
    default_gpu = is_default_gpu(args)

    # if args.random_agent:
    #     agent_class = RandomAgent
    if args.set_navigation_history:
        agent_class = GMapWithContextAgent
        print("setting agent_class to GMapWithContextAgent")
    elif args.extend_wta:
        agent_class = GMapNavAgnetWta
        print("setting agent_class to GMapNavAgnetWta")
    else:
        agent_class = GMapNavAgent
        print("setting agent_class to GMapNavAgent")

    agent = agent_class(args, val_envs["val_seen"], rank=rank)


    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))
        
    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    wandb_log = {}
    for env_name, env in val_envs.items():
        print("env_name", env_name)
        # if os.path.exists(os.path.join(args.pred_dir, "submit_%s.json" % env_name)):
        #     print("file already exists", os.path.join(args.pred_dir, "submit_%s.json" % env_name))
        #     continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        if args.random_agent:
            agent.test()
        else:
            agent.test(use_dropout=False, feedback='argmax', iters=iters)

        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output, wta_test=args.extend_wta)
        preds = merge_dist_results(all_gather(preds))
        print("preds", preds)

        if default_gpu:
            # if 'test' not in env_name:
            score_summary, metrics = env.eval_metrics(preds, wta_mode=args.extend_wta)
            loss_str = "Env name: %s" % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
                wandb_log[f"{env_name}_{metric}"] = val
            write_to_record_file(loss_str+'\n', record_file, verbose=False)


            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
                json.dump(
                    dict(metrics),
                    open(os.path.join(args.pred_dir, "metric_%s.json" % env_name), 'w'),
                    cls=NumpyEncoder,
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    if args.wandb_log:
        wandb.log(wandb_log)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def main():
    args = parse_args()
    print(args)

    if args.wandb_log:
        wandb.init(
            project=args.wandb_project,
            config=args,
            id=args.id, 
            resume=True
        )
    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)

    record_file = os.path.join(args.log_dir, 'dataset.txt')
    train_env, val_envs, aug_train_env = build_dataset(args, val_only=args.test, aug=args.aug, except_train_seen = args.except_train_seen, rank=rank, append_q=args.append_q, append_history=args.append_history, record_file=record_file)
    if not args.test:
        print("start training ... ")
        train(args, train_env, val_envs, aug_env=aug_train_env, rank=rank)
    else:
        valid(args, val_envs, rank=rank)
            

if __name__ == '__main__':
    main()