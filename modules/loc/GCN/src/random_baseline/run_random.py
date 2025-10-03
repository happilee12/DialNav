import json
import random
import numpy as np
from src.utils import evaluate
from src.cfg import *


def random_node_selection(args, split):
    splitData = json.load(open(args.data_dir + split))
    print("splitData", args.data_dir + split)
    submission = {}
    for episode in splitData:
        nodes = [n for n in args.scan_graphs[episode["scan"]].nodes()]
        # nodes = [n for n in args.scan_graphs[episode["scanName"]].nodes()]
        vp = random.choice(nodes)
        submission[episode["instr_id"]] = {"viewpoint": vp}
        # submission[episode["episodeId"]] = {"viewpoint": vp}
    fileName = (
        args.predictions_dir
        + "/randomBaseline_"
        + split.split(".")[0]
        + "_submission.json"
    )
    print(fileName)
    json.dump(
        submission,
        open(fileName, "w"),
        indent=3,
    )
    return fileName


if __name__ == "__main__":
    args = parse_args()

    data_splits = [
        # "train_data.json",
        # "valSeen_data.json",
        # "valUnseen_data.json",
        # "test_data_full.json",
        # "train.json",
        "val_seen.json",
        "val_unseen.json",
        "test.json",
    ]

    for splitFile in data_splits:
        filename = random_node_selection(
            args,
            splitFile,
        )
        evaluate(args, args.data_dir + splitFile, split_name=splitFile.split(".")[0], run_name="randomBaseline")
