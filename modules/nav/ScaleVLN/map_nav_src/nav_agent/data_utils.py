import os
import json
import random
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_bert(text):
    return tokenizer.encode(text)

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if tokenizer == 'bert':
                # if file exists
                jsonfilepath = os.path.join(anno_dir, '%s.json' % (split))
                jsonlfilepath = os.path.join(anno_dir, '%s.jsonl' % (split))
                if os.path.exists(jsonlfilepath):
                    new_data = []
                    with open(jsonlfilepath) as f:
                        for line in f:
                            new_data.append(json.loads(line))
                else:
                    with open(jsonfilepath) as f:
                        new_data = json.load(f)
                data += new_data
            else:
                raise NotImplementedError('unsupported tokenizer %s' % tokenizer)

        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True, append_q=False, append_history=False, maximum_navigation_history_length=None):
    data = []
    instr_encoding_lengths = []
    for item in load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test):
        ## For vln baseline training
        # if item['_chat_idx'] > 0:
        #     print("using only first item")
        #     continue
        item['path_id'] = f"{item['instr_id']}"

        ### instruction
        instruction = "target : "+item['target']
        target_only_instruction = "target : "+item['target']
        if 'a' in item:
            instruction = f"a: {item['a']} " + instruction
        if append_q and 'q' in item:
            instruction = f"q: {item['q']} " + instruction
        if append_history and item['_chat_idx'] > 1:
            dialog_history = ''
            for dialog in item['_full_dialog'][:item['_chat_idx']-1]:
                dialog_history = dialog_history + f"q: {dialog['q']} a: {dialog['a']} "
            instruction = dialog_history + instruction
        instr_encoding = encode_bert(instruction)
        target_only_instr_encoding = encode_bert(target_only_instruction)
        instr_encoding_lengths.append(len(instr_encoding))
        item['instr_encoding'] = instr_encoding[-max_instr_len:]
        item['instruction'] = instruction
        item['target_only_instruction'] = target_only_instruction
        item['target_only_instr_encoding'] = target_only_instr_encoding


        # for training with dialog history
        if maximum_navigation_history_length is not None:
            if len(item['nav_history']) > maximum_navigation_history_length:
                continue
        if '_full_dialog' in item and 'nav_history' in item:
            item['context'] = {
                '_full_dialog': item['_full_dialog'],
                'nav_history': item['nav_history'],
            }

        data.append(item)
    
    metadata = {}
    metadata['count'] = len(data)
    metadata['sample'] = random.choice(data)
    metadata['instr_encoding_length_qurtile'] = np.percentile(instr_encoding_lengths, [25, 50, 75])
    instr_encoding_lengths = [len(item['instr_encoding']) for item in data]
    metadata['cut_instr_encoding_length_qurtile'] = np.percentile(instr_encoding_lengths, [25, 50, 75])

    return data, metadata
