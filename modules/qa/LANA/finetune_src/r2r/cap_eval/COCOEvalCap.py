__author__ = 'tylin'
from importlib.resources import path
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
import networkx as nx
import sys
from collections import defaultdict
from .tokenizer.ptbtokenizer import PTBTokenizer

class COCOEvalCap(object):
    def __init__(self, val_instr_data):
    # def __init__(self, splits, tok, val_instr_data, use_clip16=False):
        self.evalImgs = []
        self.eval = {}
        # self.splits = splits
        self.gts = defaultdict(list)
        # self.use_clip16 = use_clip16
        for item in val_instr_data:
            self.gts[str(item['path_id'])].append(item['instruction'])


    def evaluate(self, path2inst):
        for k,v in path2inst.items():
            path2inst[k] = [v]

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(self.gts)
        res = tokenizer.tokenize(path2inst)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    # self.setImgToEvalImgs(scs, gts.keys(), m)
                    # print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                # self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.3f"%(method, score))
        #self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    # def setImgToEvalImgs(self, scores, imgIds, method):
    #     for imgId, score in zip(imgIds, scores):
    #         if not imgId in self.imgToEval:
    #             self.imgToEval[imgId] = {}
    #             self.imgToEval[imgId]["image_id"] = imgId
    #         self.imgToEval[imgId][method] = score

    # def setEvalImgs(self):
    #     self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]