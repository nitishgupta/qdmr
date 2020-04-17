from typing import List, Dict, Set

import os
import random
import argparse
from collections import defaultdict

from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, convert_nestedexpr_to_tuple

from qdmr.data import utils

random.seed(1)

""" Split the original Break-annotation into train/dev/test. The original data only comes with train & dev """


def select_train_templates(target_train_num_ques: int, template_list: List[str],
                           template2count: Dict[str, int]):
    train_templates = []
    train_num_ques = 0

    random.shuffle(template_list)
    template_num = 0
    while train_num_ques < target_train_num_ques:
        template = template_list[template_num]
        train_templates.append(template)
        train_num_ques += template2count[template]
        template_num += 1

    return train_templates, train_num_ques


def funcs_in_templates(templates: List[str], template2functions: Dict[str, Set[str]]):
    func_set = set()
    for template in templates:
        func_set.update(template2functions[template])
    return func_set


def template_based_split(train_qdmr_examples: List[QDMRExample], dev_qdmr_examples: List[QDMRExample],
                         train_ratio: float):
    """ This data is processed using parse_dataset.parse_qdmr and keys in this json can be glanced at from there.

    The nested_expression in the data makes life easier since functions are already normalized to their min/max, add/sub
    identifier.

    train_ratio: ``float`` Is the ratio of train_ques to total questions available
    """

    # Merge original train and dev to make new splits
    qdmr_examples = train_qdmr_examples + dev_qdmr_examples

    qid2qdmrexample = {}
    for qdmr_example in qdmr_examples:
        qid2qdmrexample[qdmr_example.query_id] = qdmr_example

    all_functions_set = set()
    template2count = defaultdict(int)
    template2qids = defaultdict(list)
    template2functions = defaultdict(set)
    for qdmr_example in qdmr_examples:
        func_set, template = convert_nestedexpr_to_tuple(qdmr_example.typed_masked_nested_expr)
        all_functions_set.update(func_set)
        template2functions[template].update(func_set)
        template2count[template] += 1
        template2qids[template].append(qdmr_example.query_id)

    total_num_ques = len(qdmr_examples)
    target_train_num_ques = int(train_ratio * total_num_ques)
    template_list = list(template2qids.keys())

    print("Number of total functions: {}".format(len(all_functions_set)))
    print("Total questions: {} Target num train ques: {}".format(total_num_ques, target_train_num_ques))
    print("Total number of templates: {}".format(len(template_list)))

    # Trying multiple splits, will choose the one with
    # 1. Num of train questions with tolerance_num_train_ques=200 ques of target
    # 2. All functions in train templates
    # 3. Max number of functions in dev (test) set
    num_chances = 1000
    all_train_splits: List[List[str]] = []
    closest_split_to_target = 0
    max_dev_numfuncs = 0
    tolerance_num_train_ques = 200
    for i in range(num_chances):
        tr_templates, num_tr_ques = select_train_templates(target_train_num_ques=target_train_num_ques,
                                                           template_list=template_list,
                                                           template2count=template2count)
        # Set of functions in this train-template split
        tr_func_set: Set[str] = funcs_in_templates(templates=tr_templates, template2functions=template2functions)
        if len(tr_func_set) < len(all_functions_set):
            continue
        test_templates = list(set(template2count.keys()).difference(set(tr_templates)))
        test_func_set: Set[str] = funcs_in_templates(templates=test_templates, template2functions=template2functions)
        num_dev_funcs = len(test_func_set)

        all_train_splits.append(tr_templates)
        diff_to_target = abs(target_train_num_ques - num_tr_ques)
        if diff_to_target <= tolerance_num_train_ques:
            if num_dev_funcs > max_dev_numfuncs:
                closest_split_to_target = len(all_train_splits) - 1  # Cannot use "i" since not all splits are valid
                max_dev_numfuncs = num_dev_funcs

    train_templates = all_train_splits[closest_split_to_target]
    num_train_ques = sum([template2count[x] for x in train_templates])
    tr_func_set: Set[str] = funcs_in_templates(templates=train_templates, template2functions=template2functions)

    print("Number of train questions: {}".format(num_train_ques))
    print("Number of train templates: {}".format(len(train_templates)))
    print("Number of total functions in train: {}".format(len(tr_func_set)))

    test_templates = list(set(template2count.keys()).difference(set(train_templates)))
    test_func_set: Set[str] = funcs_in_templates(templates=test_templates, template2functions=template2functions)
    print("Number of test templates: {}".format(len(test_templates)))
    print("Number of test functions: {}".format(len(test_func_set)))

    print("Function not in test: {}".format(tr_func_set.difference(test_func_set)))

    train_qids = []
    train_qdmrs = []
    # In-domain dev set
    dev_in_qids = []
    dev_in_qdmrs = []
    train_dev_in_ratio = 0.1
    for t in train_templates:
        t_qids = template2qids[t]
        # Split qids for this template into 0.9/0.1 for train/dev-in
        t_dev_qids = t_qids[0:int(train_dev_in_ratio * len(t_qids))]
        t_train_qids = t_qids[int(train_dev_in_ratio * len(t_qids)):]
        train_qids.extend(t_train_qids)
        dev_in_qids.extend(t_dev_qids)
        train_qdmrs.extend([qid2qdmrexample[qid] for qid in t_train_qids])
        dev_in_qdmrs.extend([qid2qdmrexample[qid] for qid in t_dev_qids])

    test_qids = []
    test_qdmrs = []
    # Out-of-domain dev set
    dev_out_qids = []
    dev_out_qdmrs = []
    test_dev_out_ratio = 0.15
    for t in test_templates:
        t_qids = template2qids[t]
        # Split qids for this template into 0.1/0.9 for dev-out/test
        t_dev_qids = t_qids[0:int(test_dev_out_ratio * len(t_qids))]
        t_test_qids = t_qids[int(test_dev_out_ratio * len(t_qids)):]
        dev_out_qids.extend(t_dev_qids)
        test_qids.extend(t_test_qids)
        dev_out_qdmrs.extend([qid2qdmrexample[qid] for qid in t_dev_qids])
        test_qdmrs.extend([qid2qdmrexample[qid] for qid in t_test_qids])

    print(f"Number of questions; train:{len(train_qdmrs)} dev-in:{len(dev_in_qdmrs)} "
          f"dev-out:{len(dev_out_qdmrs)} test:{len(test_qdmrs)}")

    return train_qdmrs, dev_in_qdmrs, dev_out_qdmrs, test_qdmrs


def main(args):
    train_qdmr_json = args.train_qdmr_json
    dev_qdmr_json = args.dev_qdmr_json

    std_split_dir = args.std_split_dir

    train_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(train_qdmr_json)
    dev_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(dev_qdmr_json)


    # Train/dev ratio - choosen based on DROP
    train_dev_ratio = 0.1
    random.shuffle(train_qdmr_examples)
    std_train_qdmrs = train_qdmr_examples[int(train_dev_ratio * len(train_qdmr_examples)):]
    std_dev_qdmrs = train_qdmr_examples[0:int(train_dev_ratio * len(train_qdmr_examples))]
    # We're making the BREAK dev set as standard-split's test
    std_test_qdmrs = dev_qdmr_examples
    print("Standard-split  Train: {}  Dev: {}  Test: {}".format(len(std_train_qdmrs), len(std_dev_qdmrs),
                                                                len(std_test_qdmrs)))

    print("Writing output to: {}".format(std_split_dir))
    if not os.path.exists(std_split_dir):
        os.makedirs(std_split_dir, exist_ok=True)

    utils.write_qdmr_examples_to_json(qdmr_examples=std_train_qdmrs,
                                      qdmr_json=os.path.join(std_split_dir, "train.json"))

    utils.write_qdmr_examples_to_json(qdmr_examples=std_dev_qdmrs,
                                      qdmr_json=os.path.join(std_split_dir, "dev.json"))

    utils.write_qdmr_examples_to_json(qdmr_examples=std_test_qdmrs,
                                      qdmr_json=os.path.join(std_split_dir, "test.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_qdmr_json", required=True)
    parser.add_argument("--dev_qdmr_json", required=True)
    parser.add_argument("--std_split_dir", required=True)

    args = parser.parse_args()

    main(args)
