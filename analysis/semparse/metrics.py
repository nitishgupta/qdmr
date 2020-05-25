import os
import json


ROOT_DATA_DIR = "/shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/splits"

splits = ["drop-template-manual", "drop-template-manual-ds"]  # "drop-standard", "drop-standard-ds"

def print_metrics(split, attn):
    # model_dir = "/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/new-splits/{}/Seq2Seq-glove/BS_4/INORDER_true/ATTNLOSS_{}/S_1337_D".format(split, attn)
    model_dir = "/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/new-splits/{}/Grammar-glove/BS_4/ATTNLOSS_{}/S_1337".format(split, attn)

    print()
    print(model_dir)
    print("SPLIT: {}  Attn:{}".format(split, attn))

    metrics_file = os.path.join(model_dir, "metrics.json")
    try:
        train_metrics = json.load(open(metrics_file))
        print("Best epoch: {}".format(train_metrics["best_epoch"]))
    except FileNotFoundError:
        print("Not found: {}".format(metrics_file))

    pred_dir = os.path.join(model_dir, "predictions")
    try:
        dev_metrics = json.load(open(pred_dir + "/dev_metrics.json"))
        print("Dev-metrics: {}".format(dev_metrics))
    except FileNotFoundError:
        print("Not found: {}".format(pred_dir + "/dev_metrics.json"))

    try:
        test_metrics = json.load(open(pred_dir + "/test_metrics.json"))
        print("Test-metrics: {}".format(test_metrics))
    except FileNotFoundError:
        print("Not found: {}".format(pred_dir + "/test_metrics.json"))


for split in splits:
    for attn in ["false", "true"]:
    # for attn in ["true"]:
        print_metrics(split, attn)




