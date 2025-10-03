import argparse
from ast import arg
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--id', type=str, default='default_project')
    parser.add_argument('--seed', type=int, default=0)


    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")

    ### model
    parser.add_argument('--max_instr_len', type=int, default=512)
    

    parser.add_argument('--basepath', type=str, default='')
    
    ### navigation data
    parser.add_argument('--connectivity_dir', type=str, default='')

    ### environment
    parser.add_argument('--env_names', type=str, default='val_seen,val_unseen,test')
    parser.add_argument('--val_seen_anno_paths', type=str, default='')
    parser.add_argument('--val_unseen_anno_paths', type=str, default='')
    parser.add_argument('--test_anno_paths', type=str, default='')

    ### batch size
    parser.add_argument('--batch_size', type=int, default=8)

    ## navigation setup
    parser.add_argument('--max_action_len', type=int, default=50)
    parser.add_argument('--mode', type=str, default='navonly')

    ### output path
    parser.add_argument('--output_path', type=str, default='./output')

    # Exp Setups
    parser.add_argument('--nav_resume_file', type=str, default='')
    parser.add_argument('--qg_resume_file', type=str, default='')
    parser.add_argument('--ag_resume_file', type=str, default='')
    parser.add_argument('--loc_resume_file', type=str, default='')
    parser.add_argument('--wta_mode', type=str, default='navigation_model')
    parser.add_argument('--nav_model', type=str, default='ScaleVLN', choices=['ScaleVLN', 'DialogHistoryAgent'])
    parser.add_argument('--loc_model', type=str, default='DuetLoc', choices=['GCN', 'DuetLoc'])
    parser.add_argument('--ag_eot_token', action='store_true', default=False)
    parser.add_argument('--nav_prob_mode', type=str, default='navigation_dialog_history', choices=['navigation', 'navigation_dialog_history'])
    parser.add_argument('--nav_wta_question_threshold', type=float, default=0.5)

    ### nav options
    parser.add_argument('--nav_act_visited_nodes', action='store_true', default=False)
    
    ### loc options
    parser.add_argument('--loc_node_feats_dir', type=str, default='')
    parser.add_argument('--loc_geodistance_nodes_path', type=str, default='')
    parser.add_argument('--loc_embedding_dir', type=str, default='')
    parser.add_argument('--loc_bert_enc', action='store_true', default=False)
    
    ### lana options
    parser.add_argument('--qa_clip_tokenizer_path', type=str, default='')

    ### wandb
    parser.add_argument('--wandb_project', type=str, default='DialNav-Holistic')
    parser.add_argument('--wandb_log', action='store_true', default=False)

    
    return parser.parse_args()