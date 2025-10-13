import json

def load_instr_datasets(anno_paths):
    data = []
    for anno_path in anno_paths:
        with open(anno_path) as f:
            new_data = json.load(f)
        data += new_data
    return data

def construct_instrs(anno_paths, tokenizer, max_instr_len=512):
    data = []
    for item in load_instr_datasets(anno_paths):
        item['path_id'] = f"{item['instr_id']}"
        instruction = "target : "+item['target']
        instr_encoding = tokenizer.encode(instruction)
        item['instr_encoding'] = instr_encoding[-max_instr_len:]
        item['instruction'] = instruction
        item['path'] = item['nav_steps']
        item['heading'] = 3.14
        data.append(item)
    return data
