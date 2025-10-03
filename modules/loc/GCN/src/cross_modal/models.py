import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel

class AttentionModel(nn.Module):
    def __init__(self, opt):
        super(AttentionModel, self).__init__()
        # Create Models
        self.txt_encoder = EncoderTextAttn(opt)
        self.img_encoder = EncoderImageAttn(opt)
        self.predict = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )
        self.img_encoder.img_lin1.apply(self.init_weights)
        self.predict.apply(self.init_weights)

    def forward(self, node_feats, text, seq_length):
        """Compute the image and caption embeddings"""
        cap_emb = self.txt_encoder(text, seq_length)
        img_emb = self.img_encoder(node_feats, cap_emb)
        joint_emb = torch.mul(
            img_emb,
            cap_emb.unsqueeze(1).expand(
                node_feats.size()[0], node_feats.size()[1], 2048
            ),
        )
        joint_emb = self.predict(joint_emb)
        predict = F.log_softmax(joint_emb, 1)
        return predict

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class EncoderImageAttn(nn.Module):
    def __init__(self, opt):
        super(EncoderImageAttn, self).__init__()
        self.img_lin1 = nn.Linear(opt.pano_embed_size, 2048)
        # self.img_lin2 = nn.Linear(opt.pano_embed_size, 1024)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images, cap_emb):
        """self attention"""

        """dialog attention"""
        img_emb = self.img_lin1(images)
        batch, pano, patch, feat = img_emb.size()
        img_emb = img_emb.view(batch * pano, patch, feat)
        cap_emb = torch.repeat_interleave(cap_emb, pano, dim=0)
        attention = self.softmax(torch.bmm(img_emb, cap_emb.unsqueeze(2)))
        img_emb = torch.bmm(attention.permute(0, 2, 1), img_emb)
        img_emb = img_emb.view(batch, pano, feat)
        return img_emb


class EncoderTextAttn(nn.Module):
    def __init__(self, opt):
        super(EncoderTextAttn, self).__init__()
        self.bidirectional = opt.bidirectional
        # self.input_size = opt.rnn_input_size
        self.hidden_size = opt.rnn_hidden_size
        self.embed_size = opt.rnn_embed_size
        self.reduce = "last" if not opt.bidirectional else "mean"
        self.embedding_dir = opt.embedding_dir
        self.bert_enc = opt.bert_enc

        if not self.bert_enc:
            glove_weights = torch.FloatTensor(
                np.load(self.embedding_dir + "glove_weights_matrix.npy", allow_pickle=True)
            )
            # self.embedding = nn.Embedding(self.input_size, self.embed_size)
            self.embedding = nn.Embedding(len(glove_weights), self.embed_size)
            self.embedding.from_pretrained(glove_weights)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert.eval()
            self.embed_size = 768  # BERT hidden size
            for param in self.bert.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            self.embed_size,
            1024,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.0,
            num_layers=1,  # self.num_layers,
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, seq_lengths):
        if self.bert_enc:
            attention_mask = (x != 0).float()  # 0은 BERT의 PAD token id
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=x.to(self.bert.device),
                    attention_mask=attention_mask.to(self.bert.device)
                )
                embed = outputs.last_hidden_state
                embed = self.dropout(embed)
                
            embed_packed = pack_padded_sequence(
                embed, seq_lengths.cpu(), enforce_sorted=False, batch_first=True
            )
        else:
            embed = self.embedding(x)
            embed = self.dropout(embed)
            embed_packed = pack_padded_sequence(
                embed, seq_lengths.cpu(), enforce_sorted=False, batch_first=True
            )

        out_packed = embed_packed
        self.lstm.flatten_parameters()
        out_packed, _ = self.lstm(out_packed)
        out, _ = pad_packed_sequence(out_packed)

        # reduce the dimension
        if self.reduce == "last":
            out = out[seq_lengths - 1, np.arange(len(seq_lengths)), :]
        elif self.reduce == "mean":
            seq_lengths_ = seq_lengths.unsqueeze(-1)
            out = torch.sum(out[:, np.arange(len(seq_lengths_)), :], 0) / seq_lengths_

        return out

##################### MODEL 2 #####################
"""BASIC MODEL"""


class BasicModel(nn.Module):
    def __init__(self, opt):
        super(BasicModel, self).__init__()
        # Create Models
        self.txt_encoder = EncoderText(opt)
        self.img_encoder = EncoderImage(opt)

    def forward(self, node_feats, text, seq_length):
        """Compute the image and caption embeddings"""
        cap_emb = self.txt_encoder(text, seq_length)
        img_emb = self.img_encoder(node_feats, cap_emb)
        return img_emb


class EncoderImage(nn.Module):
    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.ffc = nn.Sequential(
            nn.Linear(opt.pano_embed_size, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
        )
        self.predict = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )
        self.sig = nn.Sigmoid()
        self.ffc.apply(self.init_weights)
        self.predict.apply(self.init_weights)

    def forward(self, images, cap_emb):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        batch_size, num_nodes, patches, feats = images.size()
        features = self.ffc(images)  # (N,340,36,64)
        features = features.flatten(2)  # (N,340,2304)
        feat_size = cap_emb.size()[1]
        cap_emb = cap_emb.unsqueeze(1).expand(
            batch_size, num_nodes, feat_size
        )  # (N,340,cap_embed_size)
        features = torch.mul(features, cap_emb)
        predict = self.predict(features)  # (N,340,2560)
        predict = F.log_softmax(predict, 1)
        return predict

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class EncoderText(nn.Module):
    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.bidirectional = opt.bidirectional
        # self.input_size = opt.rnn_input_size
        self.hidden_size = opt.rnn_hidden_size
        self.reduce = "last" if not opt.bidirectional else "mean"
        self.embedding_dir = opt.embedding_dir
        self.bert_enc = opt.bert_enc

        if not self.bert_enc:
            glove_weights = torch.FloatTensor(
                np.load(self.embedding_dir + "glove_weights_matrix.npy", allow_pickle=True)
            )
            # self.embedding = nn.Embedding(self.input_size, self.embed_size)
            self.embedding = nn.Embedding(len(glove_weights), self.embed_size)
            self.embedding.from_pretrained(glove_weights)
            self.embed_size = opt.rnn_embed_size
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert.eval()
            self.embed_size = 768  # BERT hidden size
            for param in self.bert.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            self.embed_size,
            1152,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.0,
            num_layers=1,
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, seq_lengths):
        if self.bert_enc:
            attention_mask = (x != 0).float()  # 0 is the PAD token id for BERT
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=x.to(self.bert.device),
                    attention_mask=attention_mask.to(self.bert.device)
                )
                embed = outputs.last_hidden_state
                embed = self.dropout(embed)
                
            embed_packed = pack_padded_sequence(
                embed, seq_lengths.cpu(), enforce_sorted=False, batch_first=True
            )
        else:
            embed = self.embedding(x)
            embed = self.dropout(embed)
            embed_packed = pack_padded_sequence(
                embed, seq_lengths.cpu(), enforce_sorted=False, batch_first=True
            )

        out_packed = embed_packed
        self.lstm.flatten_parameters()
        out_packed, _ = self.lstm(out_packed)
        out, _ = pad_packed_sequence(out_packed)

        # reduce the dimension
        if self.reduce == "last":
            out = out[seq_lengths - 1, np.arange(len(seq_lengths)), :]
        elif self.reduce == "mean":
            seq_lengths_ = seq_lengths.unsqueeze(-1)
            out = torch.sum(out[:, np.arange(len(seq_lengths_)), :], 0) / seq_lengths_

        return out





from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class ResidualGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualGCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        residual = x
        out = self.gcn(x, edge_index)
        out = self.norm(out)
        out = self.relu(out + residual)
        return out


class GCNAttentionModel(nn.Module):
    def __init__(self, opt):
        super(GCNAttentionModel, self).__init__()
        
        self.txt_encoder = EncoderTextAttn(opt)
        self.img_encoder = EncoderImageAttn(opt)
        
        subviews = 36
        input_dim = 2048
        self.hidden_dim = 2048

        self.gcn_layers = nn.ModuleList()
        for i in range(opt.num_gcn_layers):
            if i == 0:
                self.gcn_layers.append(ResidualGCNLayer(input_dim, self.hidden_dim)) # input: 2048
            elif i == opt.num_gcn_layers - 1:
                self.gcn_layers.append(ResidualGCNLayer(self.hidden_dim, self.hidden_dim))
            else:
                self.gcn_layers.append(ResidualGCNLayer(self.hidden_dim, 2048))
        
        self.predict = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )
        
        self.img_encoder.img_lin1.apply(self.init_weights)
        self.predict.apply(self.init_weights)

    def create_data_list(self, node_feats, edge_index):
        data_list = []
        batch_size = node_feats.size(0)

        for i in range(batch_size):
            node_feat = node_feats[i]           # (num_nodes, feature_dim)
            edge_idx = edge_index[i]            # (2, num_edges)

            data = Data(x=node_feat, edge_index=edge_idx)
            data_list.append(data)

        return data_list

    def forward(self, node_feats, text, seq_length, edge_index):
        cap_emb = self.txt_encoder(text, seq_length)
        x = node_feats
        data_list = self.create_data_list(node_feats, edge_index)
        batch = Batch.from_data_list(data_list) 

        edge_index, batch_idx = batch.edge_index, batch.batch

        x  = torch.mean(batch.x, dim=1)

        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
            x = F.relu(x)

        batch_size = cap_emb.shape[0] 
        num_nodes_per_graph = 345 
        num_features = 2048 
        x = x.view(batch_size, num_nodes_per_graph, 2048)

        cap_emb = cap_emb.unsqueeze(1)
        joint_emb = x * cap_emb 
        joint_emb = self.predict(joint_emb)
        predict = F.log_softmax(joint_emb, 1)
        return predict

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)