BASEPATH=PATH_TO_THE_PROJECT
OUTPUT_PATH=OUTPUT_PATH

CONNECTIVITY_DIR=${BASEPATH}/dataset/connectivity/
VAL_SEEN_ANNO_PATHS=${BASEPATH}/dataset/RAIN/holistic/val_seen.json
VAL_UNSEEN_ANNO_PATHS=${BASEPATH}/dataset/RAIN/holistic/val_unseen.json
LOC_NODE_FEATS_DIR=${BASEPATH}/dataset/modules/node_feats/
QA_CLIP_TOKENIZER_PATH=${BASEPATH}/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz
LOC_GEODISTANCE_NODES_PATH=${BASEPATH}/dataset/modules/localization/geodistance_nodes.json
LOC_EMBEDDING_DIR=${BASEPATH}/dataset/modules/localization/word_embeddings/
# op
gpu=0
wta_mode='ct_0.9'
nav=${BASEPATH}/dataset/checkpoints/ScaleVLN
lanaa=${BASEPATH}/dataset/checkpoints/lana_a
lanaq=${BASEPATH}/dataset/checkpoints/lana_q
loc=${BASEPATH}/dataset/checkpoints/gcn.pt

# options
default_options="--basepath ${BASEPATH} 
--connectivity_dir ${CONNECTIVITY_DIR}
--val_seen_anno_paths ${VAL_SEEN_ANNO_PATHS}
--val_unseen_anno_paths ${VAL_UNSEEN_ANNO_PATHS}
--env_names val_seen,val_unseen
--loc_node_feats_dir ${LOC_NODE_FEATS_DIR}
--qa_clip_tokenizer_path ${QA_CLIP_TOKENIZER_PATH}
--loc_geodistance_nodes_path ${LOC_GEODISTANCE_NODES_PATH}
--loc_embedding_dir ${LOC_EMBEDDING_DIR}
--nav_wta_question_threshold 0.7
"

holistic_options="--mode holistic"
gtloc_options="--mode gt_loc"
# wandb_options="--wandb_project DialNav-Holistic --wandb_log"
wandb_options=""


project_id=holistic
echo "running ${project_id}"
model_paths="--nav_resume_file ${nav} 
--ag_resume_file ${lanaa} 
--qg_resume_file ${lanaq} 
--loc_resume_file ${loc}"
OUTPUT_PATH=${OUTPUT_PATH}/${project_id}
CUDA_VISIBLE_DEVICES=${gpu} python3 main.py ${default_options} --id ${project_id} --output_path ${OUTPUT_PATH} ${holistic_options} ${wandb_options} --wta_mode ${wta_mode} ${model_paths} --nav_model ScaleVLN --loc_model GCN

project_id=gtloc
echo "running ${project_id}"
model_paths="--nav_resume_file ${nav} 
--ag_resume_file ${lanaa} 
--qg_resume_file ${lanaq} 
--loc_resume_file ${loc}"
OUTPUT_PATH=${OUTPUT_PATH}/${project_id}
CUDA_VISIBLE_DEVICES=${gpu} python3 main.py ${default_options} --id ${project_id} --output_path ${OUTPUT_PATH} ${gtloc_options} ${wandb_options} --wta_mode ${wta_mode} ${model_paths} --loc_model GCN --nav_model ScaleVLN

