import argparse
import json
from Llava.llava_questioner import LLAVAQuesetioner
import os

def generate_questions_for_rain(target_file, output_file, question_module):
    with open(target_file, 'r') as f:
        data = json.load(f)
    

    for item in data:
        item['org_q'] = item["q"]
        q, seen_paths = question_module.say([item['scan']], [item['start_pano']], [])
        item['q'] = q[0]
    with open(output_file, 'w') as f:
        json.dump(data, f)


### parse args


### get output path from script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=str, default=".")
    parser.add_argument("--output_path", type=str, default=".")
    args = parser.parse_args()
    question_module = LLAVAQuesetioner(args.basepath + "/captions/llava_naive.json")

    basepath = args.basepath

    envs = ['val_seen', 'val_unseen', 'test']
    output_dir = f"{args.output_path}/updated_questions"
    os.makedirs(output_dir, exist_ok=True)

    for env in envs:
        target_file = f"{basepath}/dataset/with_dialog/{env}.json"
        output_file = f"{output_dir}/{env}.json"
        generate_questions_for_rain(target_file, output_file, question_module)
