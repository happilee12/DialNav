import numpy as np
import json
from nltk.tokenize import word_tokenize
import numpy as np
import copy
import re
import csv
import sys
import numpy as np
import os
import torch
from transformers import BertTokenizer, BertModel

csv.field_size_limit(sys.maxsize)
from src.cross_modal.led_dataset import LEDDataset
from src.utils import open_graph

class Loader:
    def __init__(self,  args):
        self.datasets = {}
        self.max_length = 0
        self.args = args
        self.word2idx = {}
        self.idx2word = {}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.connectivity_lists = {}
        max_nodes = 0
        for idx, filename in enumerate(os.listdir(self.args.connect_dir)):
            if len(filename.split("_")) > 1 and filename.split("_")[1] == "connectivity.json":
                # print("load .. ", idx, filename)
                scan_id = filename.split("_")[0]
                g = open_graph(self.args.connect_dir, scan_id)
                nodes = sorted([n for n in g.nodes() ])
                # print("scanid, nodes", scan_id, nodes)

                with open(self.args.connect_dir+filename, 'r') as f:
                    connectivity_json = json.load(f)
                valid_node_idx = [True if node['image_id'] in nodes else False for idx, node in enumerate(connectivity_json)]
                
                assert len(connectivity_json) == len(valid_node_idx), "mask and list mistmatch"
                connectivity_json = [d for d, m in zip(connectivity_json, valid_node_idx) if m]
                connectivity = []
                for i, item in enumerate(connectivity_json):
                    assert len(connectivity_json[i]["unobstructed"]) == len(valid_node_idx), "mask and list mistmatch2"
                    c = [d for d, m in zip(connectivity_json[i]["unobstructed"], valid_node_idx) if m] 
                    connectivity.append(c)

                edge_index, graph_nodes = self.create_edge_index(connectivity)
                # print("scan_id", scan_id, len(nodes), graph_nodes)
                if graph_nodes > 345:
                    raise ValueError(f"edge_index exceeded maximum nodes")
                if max_nodes < graph_nodes:
                    max_nodes = graph_nodes
                self.connectivity_lists[scan_id] = edge_index 
        self.pad_and_transpose_connectivity_list()

    def get_pano_feats(self, scan_id):
        return torch.load(self.args.panofeat_dir + scan_id + ".pt", weights_only=True).cuda()
            
    def pad_and_transpose_connectivity_list(self):
        max_N = 0
        for scan_id, arrays in self.connectivity_lists.items():
            # print(arrays)
            max_N = max(max_N, len(arrays))
        

        padded_lists = {}
        for scan_id, arrays in self.connectivity_lists.items():
            for _ in range(len(arrays), max_N):
                arrays.append([0, 0])
            transposed = torch.tensor(arrays, dtype=torch.long).t()
            # print("scan_id, transposed", scan_id, transposed)
            padded_lists[scan_id] = transposed
        self.connectivity_lists = padded_lists 


    def create_edge_index(self, connectivity_list):
        edge_index = []
        for i, connections in enumerate(connectivity_list):
            for j, is_connected in enumerate(connections):
                if is_connected:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # 무방향 그래프를 위해 양방향 추가
        return edge_index, torch.tensor(edge_index).max().item()
 
    def bert_tokenize_text(self, text):
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
