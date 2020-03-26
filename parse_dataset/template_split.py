from typing import List, Dict, Set

import os
import random
import argparse
from collections import defaultdict

from qdmr.utils import read_qdmr_json_to_examples, QDMRExample, convert_nestedexpr_to_tuple

from qdmr import utils

random.seed(1)


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


def template_based_split(train_qdmr_examples: List[QDMRExample], dev_qdmr_examples: List[QDMRExample]):
    """ This data is processed using parse_dataset.parse_qdmr and keys in this json can be glanced at from there.

    The nested_expression in the data makes life easier since functions are already normalized to their min/max, add/sub
    identifier.

    This function mainly reads relevant
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
    target_train_num_ques = int(0.8 * total_num_ques)
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
        dev_templates = list(set(template2count.keys()).difference(set(tr_templates)))
        dev_func_set: Set[str] = funcs_in_templates(templates=dev_templates, template2functions=template2functions)
        num_dev_funcs = len(dev_func_set)

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

    dev_templates = list(set(template2count.keys()).difference(set(train_templates)))
    dev_func_set: Set[str] = funcs_in_templates(templates=dev_templates, template2functions=template2functions)
    print("Number of dev templates: {}".format(len(dev_templates)))
    print("Number of dev functions: {}".format(len(dev_func_set)))

    print("Function not in dev: {}".format(tr_func_set.difference(dev_func_set)))

    train_qids = []
    train_qdmrs = []
    for t in train_templates:
        train_qids.extend(template2qids[t])
        train_qdmrs.extend([qid2qdmrexample[qid] for qid in template2qids[t]])

    test_qids = []
    test_qdmrs = []
    for t in dev_templates:
        test_qids.extend(template2qids[t])
        test_qdmrs.extend([qid2qdmrexample[qid] for qid in template2qids[t]])

    return train_qdmrs, test_qdmrs


def main(args):
    train_qdmr_json = args.train_qdmr_json
    dev_qdmr_json = args.dev_qdmr_json

    train_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(train_qdmr_json)
    dev_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(dev_qdmr_json)

    tmp_based_train_qdmrs, tmp_based_test_qdmrs = template_based_split(train_qdmr_examples, dev_qdmr_examples)

    output_dir = os.path.split(args.tmp_split_train_qdmr_json)[0]
    print("Writing output to: {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    utils.write_qdmr_examples_to_json(qdmr_examples=tmp_based_train_qdmrs,
                                      qdmr_json=args.tmp_split_train_qdmr_json)

    utils.write_qdmr_examples_to_json(qdmr_examples=tmp_based_test_qdmrs,
                                      qdmr_json=args.tmp_split_test_qdmr_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_qdmr_json", required=True)
    parser.add_argument("--dev_qdmr_json", required=True)
    parser.add_argument("--tmp_split_train_qdmr_json")
    parser.add_argument("--tmp_split_test_qdmr_json")

    args = parser.parse_args()

    main(args)
