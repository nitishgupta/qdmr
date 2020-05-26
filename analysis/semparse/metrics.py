import os
import json


ROOT_DATA_DIR = "/shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/splits"


# standard_splits = ["drop-standard", "drop-standard-train-ds", "drop-standard-full-ds"]
# template_splits = ["drop-template", "drop-template-full-ds"]
# template_rsq_splits = ["drop-template-rsq", "drop-template-full-ds-rsq"]

# splits = template_splits # standard_splits +
splits = ["full-50", "full-40", "full-30", "full-20"]  # "full",


def print_single_metrics(pred_dir, metrics_json):
    metrics_file = os.path.join(pred_dir, metrics_json)
    try:
        metrics = json.load(open(metrics_file))
        print("{}: {}".format(metrics_json[:-5], metrics))
    except FileNotFoundError:
        print("Not found: {}".format(metrics_file))


def print_metrics(split, attn):
    # model_dir = "/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/splits/{}/Seq2Seq-glove/BS_16/INORDER_true/ATTNLOSS_{}/S_1".format(split, attn)
    model_dir = "/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/splits/{}/Grammar-glove/BS_16/ATTNLOSS_{}/S_1".format(split, attn)

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

    print_single_metrics(pred_dir, "dev_metrics.json")

    print_single_metrics(pred_dir, "indomain_skewed_test_metrics.json")

    print_single_metrics(pred_dir, "indomain_unbiased_test_metrics.json")

    print_single_metrics(pred_dir, "heldout_test_metrics.json")


for split in splits:
    # for attn in ["false", "true"]:
    for attn in ["true"]:
        print_metrics(split, attn)




