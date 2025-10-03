
BASEPATH=BASEPATH
CONNECTIVITY_DIR=${BASEPATH}/dataset/connectivity/
VAL_SEEN_ANNO_PATHS=${BASEPATH}/dataset/rain_dataset/holistic/val_seen.json
VAL_UNSEEN_ANNO_PATHS=${BASEPATH}/dataset/rain_dataset/holistic/val_unseen.json
TEST_ANNO_PATHS=${BASEPATH}/dataset/rain_dataset/holistic/test.json

EXP='expid'
OUTPUT_PATH_BASE=output_path/${EXP}
gpu=0
wta_mode='ct_0.9'
nav=nav_model_path
lanaa=lanaa_model_path
lanaq=lanaq_model_path
loc=loc_model_path


# options
default_options="--basepath ${BASEPATH} 
--connectivity_dir ${CONNECTIVITY_DIR}
--val_seen_anno_paths ${VAL_SEEN_ANNO_PATHS}
--val_unseen_anno_paths ${VAL_UNSEEN_ANNO_PATHS}
--test_anno_paths ${TEST_ANNO_PATHS}
--env_names val_seen,val_unseen
"

holistic_options="--mode holistic"
gtloc_options="--mode gt_loc"
wandb_options="--wandb_project DialNav-Holistic --wandb_log"


project_id=holistic
echo "running ${project_id}"
model_paths="--nav_resume_file ${nav} 
--ag_resume_file ${lanaa} 
--qg_resume_file ${lanaq} 
--loc_resume_file ${loc}"
OUTPUT_PATH=${OUTPUT_PATH_BASE}/${project_id}
CUDA_VISIBLE_DEVICES=${gpu} python3 main.py ${default_options} --id ${project_id} --output_path ${OUTPUT_PATH} ${holistic_options} ${wandb_options} --wta_mode ${wta_mode} ${model_paths} --nav_model ScaleVLN --loc_model GCN

project_id=gtloc
echo "running ${project_id}"
model_paths="--nav_resume_file ${nav} 
--ag_resume_file ${lanaa} 
--qg_resume_file ${lanaq} 
--loc_resume_file ${loc}"
OUTPUT_PATH=${OUTPUT_PATH_BASE}/${project_id}
CUDA_VISIBLE_DEVICES=${gpu} python3 main.py ${default_options} --id ${project_id} --output_path ${OUTPUT_PATH} ${gtloc_options} ${wandb_options} --wta_mode ${wta_mode} ${model_paths} --loc_model GCN --nav_model ScaleVLN

