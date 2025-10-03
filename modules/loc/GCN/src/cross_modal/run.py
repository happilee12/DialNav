import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import tqdm
import numpy as np
import os.path
import os
import json
from accelerate import Accelerator

from src.cross_modal.loader import Loader
from src.cross_modal.holistic_loader import Loader as HolisticLoader
from src.cross_modal.xrn import XRN
from src.cfg import *
from src.utils import evaluate
import wandb
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

class LEDAgent:
    def __init__(self, args):
        self.args = args
        current_device = torch.cuda.current_device()
        self.device = (
            torch.device(f"cuda:{current_device}")
            if torch.cuda.is_available() 
            else torch.device("cpu")
        )
        self.args.device = self.device
        self.loss_func = nn.KLDivLoss(reduction="batchmean")

        self.loader = None
        self.writer = None

        if args.train and args.model_save:
            self.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
            if not os.path.isdir(self.checkpoint_dir):
                print("Checkpoint directory under {}".format(self.checkpoint_dir))
                os.system("mkdir {}".format(self.checkpoint_dir))
                os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = None
        self.optimizer = None
        # self.node2pix = json.load(open(args.image_dir + "allScans_Node2pix.json"))
        # self.args.rnn_input_size = len(
        #     json.load(open(self.args.embedding_dir + "word2idx.json"))
        # )
        self.epoch = 0
        self.best_val_acc = -0.1
        self.patience = 0
        self.savename = ""

    def init_holistic_inference(self):
        self.load_model()
        self.args.scan_graphs = {}
        scans = [s.strip() for s in open(self.args.connect_dir + "scans.txt").readlines()]
        for scan_id in scans:
            self.args.scan_graphs[scan_id] = open_graph(self.args.connect_dir, scan_id)
        self.loader = HolisticLoader(args=self.args)

    def localize(self, texts, scan_ids):
        if self.args.bert_enc:
            texts, seq_lengths = self.loader.build_bert_vocab(texts)
        else:
            texts, seq_lengths = self.loader.build_pretrained_vocab(texts)
        texts = torch.tensor(texts)
        seq_lengths = torch.tensor(seq_lengths)
        node_names_batch = []
        for scan_id in scan_ids:
            node_names = sorted([n for n in self.args.scan_graphs[scan_id].nodes()]) 
            for _ in range(len(node_names), self.args.max_nodes):
                node_names.append("null")
            node_names_batch.append(node_names)
        
        node_feats = [self.loader.get_pano_feats(scan_id) for scan_id in scan_ids]
        node_feats = torch.stack(node_feats)
        
        edge_indexs = torch.stack([self.get_connectivity_list(scan_id) for scan_id in scan_ids])
        result = self.model.localize(texts, seq_lengths, node_feats, node_names_batch, edge_indexs)
        return result
    

    def get_connectivity_list(self, scan_id):
        if self.loader.connectivity_lists and scan_id in self.loader.connectivity_lists: 
            return self.loader.connectivity_lists[scan_id]
        if self.args.gcn:
            raise Exception("connectivity not found .. ")
        else:
            return torch.empty(2, 1)
        

    def load_model(self):
        self.model = XRN(self.args)
        if self.args.eval_ckpt:
            self.model.load_state_dict()



    def load_data(self):
        print("Loading Data...")
        self.loader = Loader(data_dir=self.args.data_dir, args=self.args)
        train_files = self.args.train_files.split(",") 
        print("train_files", train_files)
        valseen_files = [self.args.val_seen_file]
        valunseen_files = [self.args.val_unseen_file]
        test_files = [self.args.test_file]
        self.loader.build_dataset(mode='train', files=train_files)
        self.loader.build_dataset(mode='valSeen', files=valseen_files)
        self.loader.build_dataset(mode='valUnseen', files=valunseen_files)
        self.train_iterator = DataLoader(
            self.loader.datasets["train"],
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=1,
        )
        self.valseen_iterator = DataLoader(
            self.loader.datasets["valSeen"],
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=1,
        )

        self.val_unseen_iterator = DataLoader(
            self.loader.datasets["valUnseen"],
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=1,
        )
        if self.args.evaluate:
            self.loader.build_dataset(mode='test', files=test_files)
            self.test_iterator = DataLoader(
                self.loader.datasets["test"],
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=1,
            )


    def tensorboard_writer(self, mode, loss, acc_topk, le):
        le = np.asarray(le)
        acc0m = sum(le <= 0) * 1.0 / len(le)
        acc5m = sum(le <= 5) * 1.0 / len(le)
        acc10m = sum(le <= 10) * 1.0 / len(le)
        self.writer.add_scalar("Loss/" + mode, np.mean(loss), self.epoch)
        self.writer.add_scalar("LE/" + mode, np.mean(le), self.epoch)
        self.writer.add_scalar("Acc@5k/" + mode, np.mean(acc_topk), self.epoch)
        self.writer.add_scalar("Acc@0m/" + mode, acc0m, self.epoch)
        self.writer.add_scalar("Acc@5m/" + mode, acc5m, self.epoch)
        self.writer.add_scalar("Acc@10m/" + mode, acc10m, self.epoch)

    def scores(self, mode, acc_k1, acc_topk, le):
        print(f"\t{mode} Acc@1k: {np.mean(acc_k1)} Acc@5k: {np.mean(acc_topk)}")
        le = np.asarray(le)
        acc0m = sum(le <= 0) * 1.0 / len(le)
        acc3m = sum(le <= 3) * 1.0 / len(le)
        acc5m = sum(le <= 5) * 1.0 / len(le)
        acc10m = sum(le <= 10) * 1.0 / len(le)
        print(
            f"\t{mode} LE: {np.mean(le):.4f}, Acc@0m: {acc0m:.4f}, Acc@5m: {acc5m:.4f}, Acc@10m: {acc10m:.4f}"
        )
        return {
            "LE": np.mean(le),
            "Acc@0m": acc0m,
            "Acc@3m": acc3m,
            "Acc@5m": acc5m,
            "Acc@10m": acc10m,
        }

    def evaluate(self, data_iterator, mode, best_scores={}):
        print("Mode-", mode)
        self.model.val_start()
        loss = []
        acc_k1, acc_topk, le = [], [], []
        submission = {}
        for batch_data in tqdm.tqdm(data_iterator):
            k1, topk, e, l, ep = self.model.run_emb(*batch_data)
            loss.append(l.item())
            acc_k1.append(k1)
            acc_topk.append(topk)
            le.extend(e)
            for indx in ep:
                submission[indx[0]] = {"viewpoint": indx[1]}
        ret_score = self.scores(mode, acc_k1, acc_topk, le)
        if not self.args.evaluate:
            if mode == "ValUnseen" and self.args.model_save:
                for key in sorted(ret_score):
                    if key not in best_scores:
                        best_scores[key] = 1000 if key == 'LE' else 0
                    print(ret_score[key] , best_scores[key])
                    if (key == 'Acc@3m' and ret_score[key] > best_scores[key]) or (key == 'LE' and ret_score[key] < best_scores[key]) :
                        self.savename = f"{self.checkpoint_dir}/best_unseen_{key}.pt"
                        torch.save(self.model.get_state_dict(), self.savename)
                        print(f"saved at {self.savename}")
                if ret_score['Acc@3m'] > self.best_val_acc:
                    self.best_val_acc = ret_score['Acc@3m']
                    self.patience = -1
                self.patience += 1
        if self.args.evaluate:
            fileName = f"{self.args.run_name}_{mode}_submission.json"
            fileName = os.path.join(self.args.predictions_dir, fileName)
            json.dump(submission, open(fileName, "w"), indent=3)
            print("submission saved at ", fileName)
        return ret_score
    
    def train(self):
        print("\nStarting Training...")
        self.model.model, self.model.optimizer, self.train_iterator
        best_scores = {}
        for epoch in range(self.epoch, self.args.num_epoch):
            wandb_log={}
            self.epoch = epoch
            print("Epoch ", self.epoch)
            self.model.train_start()
            loss = []
            acc_k1, acc_topk, le = [], [], []
            print("Mode-", "train")
            for batch_data in tqdm.tqdm(self.train_iterator):
                k1, topk, e, l = self.model.train_emb(*batch_data)
                del batch_data
                loss.append(l.item())
                acc_k1.append(k1)
                acc_topk.append(topk)
                le.extend(e)

            print(f"\tTraining Loss: {np.mean(loss)}")
            wandb_log['loss'] = np.mean(loss) 
            wandb_log['epoch'] = epoch
            scores = self.scores("Training", acc_k1, acc_topk, le)
            for key in sorted(scores):
                wandb_log['train_'+key] = scores[key]

            scores = self.evaluate(self.val_unseen_iterator,  mode="ValUnseen", best_scores=best_scores)
            print(scores)
            for key in sorted(scores):
                wandb_log['val_unseen_'+key] = scores[key]
                if scores[key] > best_scores[key]:
                    best_scores[key] = scores[key]

            scores = self.evaluate(self.valseen_iterator, mode="ValSeen")
            print(scores)
            for key in sorted(scores):
                wandb_log['val_seen_'+key] = scores[key] 


            if self.patience > self.args.early_stopping:
                print(f"Patience Reached. Ending Training at Epoch {self.epoch}.")
                break
            
            print(wandb_log)
            if self.args.wandb_log:
                wandb.log(wandb_log)

    def run(self):
        os.makedirs(self.args.predictions_dir, exist_ok=True)


        if self.args.train:
            self.load_data()
            self.load_model()

            self.train()
            print("Training Ended...")
            print("Last Model Saved @", self.savename)

        if self.args.evaluate:
            self.load_data()
            self.load_model()

            test_scores = self.evaluate(self.test_iterator, mode="test")
            val_seen_scores = self.evaluate(self.valseen_iterator, mode="valSeen")
            val_unseen_scores = self.evaluate(self.val_unseen_iterator, mode="valUnseen")

            print("val seen scores", val_seen_scores)
            print("val unseen scores", val_unseen_scores)
            print("test scores", test_scores)

            scores = {
                "val_seen": val_seen_scores,
                "val_unseen": val_unseen_scores,
                "test": test_scores
            }
            with open(os.path.join(self.args.predictions_dir, "scores.json"), "w") as f:
                json.dump(scores, f)


if __name__ == "__main__":
    args = parse_args()
    agent = LEDAgent(args)
    if args.wandb_log:
        wandb.init(
            project="ICLR2026",
            group="GCN",
            id=args.run_name,
            config=args
        )
    
    agent.run()
