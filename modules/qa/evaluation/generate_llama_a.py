import argparse
import json
from Llama.llama_answerer import LlamaAnswerer
import os
def generate_answers_for_rain(target_file, output_file, answerer):
    with open(target_file, 'r') as f:
        data = json.load(f)
    
    from tqdm import tqdm
    
    batch_size = 8
    for i in tqdm(range(0, len(data), batch_size), desc="Generating answers"):
        batch = data[i:i+batch_size]
        scans = [item['scan'] for item in batch]
        paths = [item['gt_path'][:20] for item in batch]
        
        answers, seen_paths = answerer.say(scans, paths)
        
        for j, item in enumerate(batch):
            item['org_a'] = item["a"]
            item['a'] = answers[j]
            
    with open(output_file, 'w') as f:
        print("output_file", output_file)
        json.dump(data, f)


### get output path from script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=str, default=".")
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--huggingface_cache_dir", type=str, default="")
    args = parser.parse_args()
    answerer = LlamaAnswerer(args.basepath + "/captions/llava_captions.json", args.huggingface_cache_dir)

    basepath = args.basepath

    envs = ['val_seen', 'val_unseen', 'test']
    output_dir = f"{args.output_path}/updated_answers"
    os.makedirs(output_dir, exist_ok=True)


    for env in envs:
        target_file = f"{basepath}/dataset/with_dialog/{env}.json"
        output_file = f"{output_dir}/{env}.json"
        generate_answers_for_rain(target_file, output_file, answerer)
