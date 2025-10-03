import numpy as np
import json
from nltk.tokenize import word_tokenize
import numpy as np
import copy
import re
import csv
import sys
import numpy as np
import torch
import os
from transformers import BertTokenizer, BertModel
csv.field_size_limit(sys.maxsize)
from src.cross_modal.led_dataset import LEDDataset
from src.utils import open_graph

class Loader:
    def __init__(self, data_dir, args):
        self.data_dir = data_dir
        self.datasets = {}
        self.max_length = 0
        self.args = args
        self.word2idx = {}
        self.idx2word = {}

        self.pano_feats = {}
        self._load_pano_features()

        self.edge_index_list = {}
        if args.gcn:
            self._load_edge_index_list()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _load_edge_index_list(self):
        max_nodes = 0
        for filename in os.listdir(self.args.connect_dir):
            if filename.endswith("_connectivity.json"):
                scan_id = filename.split("_")[0]
                g = open_graph(self.args.connect_dir, scan_id)
                nodes = sorted(g.nodes())
                connectivity_json = self._load_json_file(self.args.connect_dir + filename)
                valid_node_idx = [node['image_id'] in nodes for node in connectivity_json]
                
                assert len(connectivity_json) == len(valid_node_idx), "mask and list mismatch"
                connectivity_json = [d for d, m in zip(connectivity_json, valid_node_idx) if m]
                connectivity = [self._filter_unobstructed(item, valid_node_idx) for item in connectivity_json]

                edge_index, graph_nodes = self._create_edge_index(connectivity)
                if graph_nodes > 345:
                    raise ValueError("edge_index exceeded maximum nodes")
                max_nodes = max(max_nodes, graph_nodes)
                self.edge_index_list[scan_id] = edge_index
        self._pad_and_transpose_edge_index_list()

    def _load_json_file(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def _filter_unobstructed(self, item, valid_node_idx):
        assert len(item["unobstructed"]) == len(valid_node_idx), "mask and list mismatch"
        return [d for d, m in zip(item["unobstructed"], valid_node_idx) if m]

    def _load_pano_features(self):
        print("self.args.device", self.args.device)
        for filename in os.listdir(self.args.panofeat_dir):
            if filename.endswith(".pt"):
                scan_id = filename.split(".")[0]
                self.pano_feats[scan_id] = torch.load(self.args.panofeat_dir + filename).to(self.args.device)

    def _pad_and_transpose_edge_index_list(self):
        max_N = max(len(arrays) for arrays in self.edge_index_list.values())
        for scan_id, arrays in self.edge_index_list.items():
            arrays.extend([[0, 0]] * (max_N - len(arrays)))
            self.edge_index_list[scan_id] = torch.tensor(arrays, dtype=torch.long).t()

    def _create_edge_index(self, connectivity_list):
        edge_index = [[i, j] for i, connections in enumerate(connectivity_list) for j, is_connected in enumerate(connections) if is_connected]
        edge_index.extend([[j, i] for i, j in edge_index])  # Add reverse edges for undirected graph
        return edge_index, torch.tensor(edge_index).max().item()
    

    def load_info(self, data, mode):
        scan_names = []
        episode_ids = []
        viewpoints = []
        dialogs = []

        
        for data_obj in data:
            ## way dataset
            if "scanName" in data_obj:
                # print("this is way dataset .. !")
                scan_names.append(data_obj["scanName"])
                episode_ids.append(data_obj["episodeId"])
                # if "test" in mode:
                #     print("this is test mode .. ! appending no viewpoint")
                #     viewpoints.append("")
                # else:
                viewpoints.append(data_obj["finalLocation"]["viewPoint"])
                # dialogs.append(self.add_tokens(data_obj["dialogArray"]))
                sentence = ""
                for idx, utterance in enumerate(data_obj["dialogArray"]):
                    if idx % 2 == 1:
                        sentence += utterance+" "

                dialogs.append(sentence) 
            else: ## dialNav
                # print("this is dialNav dataset .. !")
                scan_names.append(data_obj["scan"])
                episode_ids.append(data_obj["instr_id"])
                viewpoints.append(data_obj["start_pano"])
                if 'q' not in data_obj:
                    continue
                sentence = data_obj["q"]
                dialogs.append(sentence)
        return scan_names, episode_ids, viewpoints, dialogs
    

    # def add_tokens(self, message_arr):
    #     new_dialog = ""
    #     for enum, message in enumerate(message_arr):
    #         if enum % 2 == 0:
    #             new_dialog += "SOLM " + message + " EOLM "
    #         else:
    #             new_dialog += "SOOM " + message + " EOOM "
    #     return new_dialog

    def bert_tokenize_text(self, text):
        """
        Tokenize text using BERT tokenizer without embedding
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            tuple: (input_ids, seq_length)
        """
        # Clean text
        text = re.sub(r"\.\.+", ". ", text)
        
        # BERT tokenization
        encoded = self.tokenizer(text.lower(), 
                               padding=True,
                               truncation=True,
                               max_length=512,
                               return_tensors='pt')
        
        input_ids = encoded['input_ids'][0]  # Remove batch dimension
        seq_len = (input_ids != self.tokenizer.pad_token_id).sum().item()
        
        return input_ids.numpy(), seq_len

    def build_pretrained_vocab(self, texts):
        """Original word2idx based vocabulary encoding"""
        word2idx = json.load(open(self.args.embedding_dir + "word2idx.json"))
        ids = []
        seq_lengths = []
        for text in texts:
            text = re.sub(r"\.\.+", ". ", text)
            line_ids = []
            words = word_tokenize(text.lower())
            self.max_length = max(self.max_length, len(words))
            for word in words:
                if word in word2idx:
                    line_ids.append(word2idx[word])
                else:
                    line_ids.append(20)  # Add random token for words not in word2idx
            ids.append(line_ids)
            seq_lengths.append(len(words))
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        return text_ids, seq_lengths

    def build_bert_vocab(self, texts):
        """BERT tokenizer based vocabulary encoding"""
        ids = []
        seq_lengths = []
        
        for text in texts:
            input_ids, seq_len = self.bert_tokenize_text(text)
            self.max_length = max(self.max_length, seq_len)
            ids.append(input_ids)
            seq_lengths.append(seq_len)

        # Pad sequences to max_length
        padded_ids = np.array([
            np.pad(row, (0, self.max_length - len(row)), 'constant', 
                  constant_values=self.tokenizer.pad_token_id)
            for row in ids
        ])

        return padded_ids, seq_lengths

    def build_dataset(self, mode, files=[]):
        # mode = file.split("_")[0]
        print("[{}]: Loading JSON file...".format(mode))

        data_list = []
        for file in files:
            # if file ends with .jsonl, load as jsonl
            if file.endswith(".jsonl"):
                data = []
                with open(file, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
            else:
                data = json.load(open(file))
            # data = data[:10]
            num_samples = int(len(data))
            print(
                "[{}]: Using {} ({}%) samples frome {}".format(
                    mode, num_samples, num_samples / len(data) * 100, file
                )
            )

            data = [item for item in data if 'q' in item]
            print("target data length len(data)", len(data))

            data_list.extend(data)
        scan_names, episode_ids, viewpoints, dialogs = self.load_info(data, mode)
        texts = copy.deepcopy(dialogs)

        if self.args.bert_enc:
            texts, seq_lengths = self.build_bert_vocab(texts)
        else:
            texts, seq_lengths = self.build_pretrained_vocab(texts)

        print("[{}]: Building dataset...".format(mode))
        dataset = LEDDataset(
            mode,
            self.args,
            scan_names,
            episode_ids,
            viewpoints,
            texts,
            seq_lengths,
            dialogs,
            self.pano_feats,
            edge_index_list = self.edge_index_list
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))
