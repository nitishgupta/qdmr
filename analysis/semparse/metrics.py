from typing import List, Dict, Tuple
import os
import json
from collections import defaultdict
import numpy as np


def get_exact_match(pred_dir, metrics_json):
    metrics_file = os.path.join(pred_dir, metrics_json)
    metrics_key = metrics_json[:-5]
    try:
        metrics = json.load(open(metrics_file))
        return metrics_key, metrics["exact_match"]
    except FileNotFoundError:
        # print("Not found: {}".format(metrics_file))
        return None, None



def get_seq2seq_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Seq2Seq-glove/BS_16/INORDER_true/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_seq2seq_elmo_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Seq2Seq-elmo/BS_16/INORDER_true/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_seq2seq_bert_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Seq2Seq-bert/BS_16/INORDER_true/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_seq2seq_coverage_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Seq2Seq-glove-coverage/BS_16/INORDER_true/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_seq2seq_elmo_coverage_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Seq2Seq-elmo-coverage/BS_16/INORDER_true/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_seq2seq_spans_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Seq2Seq-glove-spans/BS_16/INORDER_true/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_seq2seq_elmo_spans_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Seq2Seq-elmo-spans/BS_16/INORDER_true/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_grammar_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Grammar-glove/BS_16/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_grammar_elmo_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Grammar-elmo/BS_16/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_grammar_bert_model_dir(splits_dir, split, attn, seed):
    model_dir = f"/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/{splits_dir}/{split}/Grammar-bert/BS_16/ATTNLOSS_{attn}/S_{seed}"
    return model_dir


def get_average_perc(values: List[float]) -> float:
    values = np.array(values) * 100.0
    perc = np.average(values)
    perc = np.around(perc, decimals=2)
    return float(perc)


def best_metrics(get_modeldir_fn, splits_dir, attn):
    print(f"Model dif fn: {get_modeldir_fn}")
    print(f"Splits: {splits_dir}")
    print(f"Attn: {attn}")

    dataset_metrics = {}

    splits = ["full-20"]
    for split in splits:
        # metrics for this dataset; {metrics_key: [values]}
        metrics_dict = defaultdict(list)
        best_seed = -1
        best_dev = -1
        for seed in [1, 2, 3, 4, 5]:
            model_dir = get_modeldir_fn(splits_dir, split, attn, seed)  # Model setting w/ seed
            pred_dir = os.path.join(model_dir, "predictions")
            # For this model/seed get exact_match for different test sets and add to metrics_dict
            _, dev_acc = get_exact_match(pred_dir, metrics_json="dev_metrics.json")
            if dev_acc is None:
                continue
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_seed = seed
        best_metric_dict = {}
        best_model_dir = get_modeldir_fn(splits_dir, split, attn, best_seed)  # Model setting w/ seed
        best_pred_dir = os.path.join(best_model_dir, "predictions")
        # for metrics_json in ["dev_metrics.json", "indomain_skewed_test_metrics.json",
        #                      "indomain_unbiased_test_metrics.json", "heldout_test_metrics.json"]:
        for metrics_json in ["indomain_unbiased_test_metrics.json", "heldout_test_metrics.json", "dev_metrics.json"]:
            metric_key, value = get_exact_match(best_pred_dir, metrics_json)
            value = np.around(value*100.0, decimals=1)
            best_metric_dict[metric_key] = value
        dataset_metrics[split] = best_metric_dict

    return dataset_metrics


def avg_metrics(get_modeldir_fn, splits_dir, attn):
    print(f"Model dif fn: {get_modeldir_fn}")
    print(f"Splits: {splits_dir}")
    print(f"Attn: {attn}")

    dataset_metrics = {}

    splits = ["full-20"]
    for split in splits:
        # metrics for this dataset; {metrics_key: [values]}
        metrics_dict = defaultdict(list)
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


# dataset_metrics = best_metrics(get_seq2seq_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
#
# dataset_metrics = best_metrics(get_seq2seq_model_dir, "resplits", "true")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
#

# dataset_metrics = best_metrics(get_seq2seq_elmo_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")

# dataset_metrics = best_metrics(get_seq2seq_elmo_model_dir, "resplits", "true")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
#


# dataset_metrics = best_metrics(get_seq2seq_bert_model_dir, "resplits", "true")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")

# dataset_metrics = best_metrics(get_seq2seq_coverage_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
# #
# dataset_metrics = best_metrics(get_seq2seq_elmo_coverage_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
# #
# dataset_metrics = best_metrics(get_seq2seq_spans_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")
# #
# dataset_metrics = best_metrics(get_seq2seq_elmo_spans_model_dir, "resplits", "false")
# print(json.dumps(dataset_metrics, indent=4))
# print("\n")

dataset_metrics = best_metrics(get_grammar_model_dir, "resplits", "false")
print(json.dumps(dataset_metrics, indent=4))
print("\n")
# #
dataset_metrics = best_metrics(get_grammar_elmo_model_dir, "resplits", "false")
print(json.dumps(dataset_metrics, indent=4))
print("\n")

# #
dataset_metrics = best_metrics(get_grammar_model_dir, "resplits", "true")
print(json.dumps(dataset_metrics, indent=4))
print("\n")
#
dataset_metrics = best_metrics(get_grammar_elmo_model_dir, "resplits", "true")
print(json.dumps(dataset_metrics, indent=4))
print("\n")

dataset_metrics = best_metrics(get_grammar_bert_model_dir, "resplits", "false")
print(json.dumps(dataset_metrics, indent=4))
print("\n")

dataset_metrics = avg_metrics(get_grammar_bert_model_dir, "resplits", "true")
print(json.dumps(dataset_metrics, indent=4))
print("\n")
