from typing import List, Dict, Tuple
import os
import json
from collections import defaultdict
import numpy as np


# splits = ["full-50", "full-40", "full-30", "full-20", "full"]  # "full",

# def print_single_metrics(pred_dir, metrics_json):
#     metrics_file = os.path.join(pred_dir, metrics_json)
#     try:
#         metrics = json.load(open(metrics_file))
#         metrics_key = metrics_json[:-5]
#         print("{}: {}".format(metrics_key, metrics))
#         return metrics_key, metrics["exact_match"]
#     except FileNotFoundError:
#         print("Not found: {}".format(metrics_file))
#
#
# def print_metrics(split, attn):
#     # model_dir = "/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/splits/{}/Seq2Seq-glove/BS_16/INORDER_true/ATTNLOSS_{}/S_1337".format(split, attn)
#     model_dir = "/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/splits/{}/Grammar-glove/BS_16/ATTNLOSS_{}/S_1".format(split, attn)
#
#     print()
#     print(model_dir)
#     print("SPLIT: {}  Attn:{}".format(split, attn))
#
#     metrics_file = os.path.join(model_dir, "metrics.json")
#     try:
#         train_metrics = json.load(open(metrics_file))
#         print("Best epoch: {}".format(train_metrics["best_epoch"]))
#     except FileNotFoundError:
#         print("Not found: {}".format(metrics_file))
#
#     pred_dir = os.path.join(model_dir, "predictions")
#
#     print_single_metrics(pred_dir, "dev_metrics.json")
#
#     print_single_metrics(pred_dir, "indomain_skewed_test_metrics.json")
#
#     print_single_metrics(pred_dir, "indomain_unbiased_test_metrics.json")
#
#     print_single_metrics(pred_dir, "heldout_test_metrics.json")


def get_exact_match(pred_dir, metrics_json):
    metrics_file = os.path.join(pred_dir, metrics_json)
    metrics_key = metrics_json[:-5]
    try:
        metrics = json.load(open(metrics_file))
        return metrics_key, metrics["exact_match"]
    except FileNotFoundError:
        print("Not found: {}".format(metrics_file))
        return None, None



def get_seq2seq_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Seq2Seq-glove/BS_16/INORDER_true/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_seq2seq_elmo_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Seq2Seq-elmo/BS_16/INORDER_true/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_seq2seq_coverage_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Seq2Seq-glove-coverage/BS_16/INORDER_true/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_grammar_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Grammar-glove/BS_16/ATTNLOSS_{attn}/S_{seed}"
    return model_dir

def get_grammar_elmo_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Grammar-elmo/BS_16/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_average_perc(values: List[float]) -> float:
    values = np.array(values) * 100.0
    perc = np.average(values)
    perc = np.around(perc, decimals=2)
    return float(perc)


def avg_metrics(get_modeldir_fn, splits_dir, attn):

    print(f"Model dif fn: {get_modeldir_fn}")
    print(f"Splits: {splits_dir}")
    print(f"Attn: {attn}")

    dataset_metrics = {}

    splits = ["full", "full-20", "full-50", "full-40", "full-30"]
    for split in splits:
        # metrics for this dataset; {metrics_key: [values]}
        metrics_dict = defaultdict(list)
        # for seed in [1, 2, 21, 42, 1337]:
        for seed in [1, 2, 3, 4, 5]:
            model_dir = get_modeldir_fn(splits_dir, split, attn, seed)  # Grammar or Seq2Seq or ....
            pred_dir = os.path.join(model_dir, "predictions")
            # For this model/seed get exact_match for different test sets and add to metrics_dict
            for metrics_json in ["dev_metrics.json", "indomain_skewed_test_metrics.json",
                                 "indomain_unbiased_test_metrics.json", "heldout_test_metrics.json"]:

                key, value = get_exact_match(pred_dir, metrics_json)
                if key:     # Returns None if file-not-found
                    metrics_dict[key].append(value)
        # For this model, average performance for different seeds across different test sets
        avg_metrics_dict = {key: get_average_perc(values) for key, values in metrics_dict.items()}
        dataset_metrics[split] = avg_metrics_dict

    return dataset_metrics

#
# dataset_metrics = avg_metrics(get_seq2seq_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
#
# dataset_metrics = avg_metrics(get_seq2seq_model_dir, "resplits", "true")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")

# dataset_metrics = avg_metrics(get_seq2seq_elmo_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
#
# dataset_metrics = avg_metrics(get_seq2seq_elmo_model_dir, "resplits", "true")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")

#
# dataset_metrics = avg_metrics(get_seq2seq_coverage_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
#

# dataset_metrics = avg_metrics(get_grammar_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
#
# dataset_metrics = avg_metrics(get_grammar_model_dir, "resplits", "true")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
#
#
# dataset_metrics = avg_metrics(get_grammar_elmo_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
#
dataset_metrics = avg_metrics(get_grammar_elmo_model_dir, "resplits", "true")
print(json.dumps(dataset_metrics, indent=4))
print("\n")
