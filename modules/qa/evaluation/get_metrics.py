from LANA.finetune_src.r2r.cap_eval.COCOEvalCap import COCOEvalCap
import argparse
import json

def get_metrics(val_instr_data, target_data):
    evaluator = COCOEvalCap(val_instr_data)

    evaluator.evaluate(target_data)
    return evaluator.eval


### main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=str, default=".")
    parser.add_argument("--target", type=str, default="q")
    parser.add_argument("--source_key", type=str, default="q")
    parser.add_argument("--target_data", type=str, default=".")
    parser.add_argument("--target_key", type=str, default="instr_id")
    parser.add_argument("--target_created", type=str, default="q")
    parser.add_argument("--output_path", type=str, default=".")
    args = parser.parse_args()

    basepath = args.basepath

    envs = ['val_seen', 'val_unseen', 'test']


    ### build references
    outputs = {}
    for env in envs:
        refernece_data_path = f"{basepath}/dataset/with_dialog/{env}.json"
        with open(refernece_data_path, 'r') as f:
            instruction_data = json.load(f)
        
        dataset_source= []
        for item in instruction_data:
            dataset_source.append({
                'instruction': item[args.source_key],
                'path_id': item['instr_id']
            })

        print("dataset", dataset_source)

        target_data_path = f"{args.target_data}/{env}.json"
        with open(target_data_path, 'r') as f:
            target_data = json.load(f)
        
        target_data_source = {}
        for item in target_data:
            target_data_source[item[args.target_key]] = item[args.target_created]

        ### get datasets
        output = get_metrics(dataset_source, target_data_source)
        print(env, " done")
        print(output)
        outputs[env] = output

    with open(f"{args.output_path}", 'w') as f:
        json.dump(outputs, f)
