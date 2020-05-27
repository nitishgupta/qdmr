from typing import List, Dict, Set, Tuple

import os
import random
import argparse
from enum import Enum
from collections import defaultdict

from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, convert_nestedexpr_to_tuple, \
    get_inorder_function_list_from_template, nested_expression_to_tree, nested_expression_to_lisp

from qdmr.data import utils
from analysis.qdmr_program_diversity import get_maps
from qdmr.domain_languages.drop_language import DROPLanguage
import numpy as np
from numpy.random import choice

np.random.seed(1)
random.seed(1)

manual_test_templates = [
    ('AGGREGATE_sum', ('AGGREGATE_min', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('AGGREGATE_avg', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN')),
    ('SELECT_NUM', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN')),
    ('AGGREGATE_sum', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN')),
    ('AGGREGATE_count', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('PROJECT', 'GET_QUESTION_SPAN', ('AGGREGATE_max', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('AGGREGATE_sum', ('FILTER_NUM_GT', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_NUMBER')),
    ('AGGREGATE_max', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('PROJECT', 'GET_QUESTION_SPAN', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN')),
    ('FILTER', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN')), 'GET_QUESTION_SPAN'),
    ('PROJECT', 'GET_QUESTION_SPAN', ('FILTER_NUM_GT', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_NUMBER')),
    ('ARITHMETIC_difference', ('AGGREGATE_sum', ('SELECT', 'GET_QUESTION_SPAN')), ('SELECT_NUM', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('ARITHMETIC_difference', ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')), ('SELECT_NUM', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('ARITHMETIC_sum', ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')), ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('ARITHMETIC_difference', 'SELECT_IMPLICIT_NUM', ('SELECT_NUM', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('ARITHMETIC_sum', ('AGGREGATE_sum', ('SELECT', 'GET_QUESTION_SPAN')), ('AGGREGATE_sum', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('AGGREGATE_count', ('FILTER_NUM_LT_EQ', ('FILTER_NUM_GT_EQ', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_NUMBER'), 'GET_QUESTION_NUMBER')),
    ('PROJECT', 'GET_QUESTION_SPAN', ('AGGREGATE_max', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN'))),
    ('ARITHMETIC_difference', ('AGGREGATE_avg', ('SELECT', 'GET_QUESTION_SPAN')), ('AGGREGATE_avg', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('ARITHMETIC_difference', ('SELECT_NUM', ('SELECT', 'GET_QUESTION_SPAN')), ('SELECT_NUM', ('AGGREGATE_min', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('AGGREGATE_count', ('COMPARATIVE', ('SELECT', 'GET_QUESTION_SPAN'), ('PARTIAL_SELECT_SINGLE_NUM', 'GET_QUESTION_SPAN'), ('CONDITION', 'GET_QUESTION_SPAN'))),
    ('COMPARISON_max', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN')), ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('ARITHMETIC_difference', ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')), ('AGGREGATE_count', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN'))),
    ('ARITHMETIC_difference', ('SELECT_NUM', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN'))), ('SELECT_NUM', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('ARITHMETIC_difference', ('SELECT_NUM', ('AGGREGATE_max', ('SELECT', 'GET_QUESTION_SPAN'))), ('SELECT_NUM', ('AGGREGATE_max', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('ARITHMETIC_difference', ('AGGREGATE_count', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN')), ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN'))),
    ('COMPARISON_max', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN'), ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN')),
    ('PROJECT', 'GET_QUESTION_SPAN', ('DISCARD', ('SELECT', 'GET_QUESTION_SPAN'), ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN'))),
    ('COMPARISON_count_max', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN'), ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN')),
    ('ARITHMETIC_difference', ('SELECT_NUM', ('AGGREGATE_min', ('SELECT', 'GET_QUESTION_SPAN'))), ('SELECT_NUM', ('AGGREGATE_max', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('ARITHMETIC_difference', ('SELECT_NUM', ('AGGREGATE_min', ('SELECT', 'GET_QUESTION_SPAN'))), ('SELECT_NUM', ('AGGREGATE_min', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('ARITHMETIC_sum', ('SELECT_NUM', ('AGGREGATE_min', ('SELECT', 'GET_QUESTION_SPAN'))), ('SELECT_NUM', ('AGGREGATE_min', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('ARITHMETIC_sum', ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')), ('ARITHMETIC_sum', ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')), ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('ARITHMETIC_sum', ('SELECT_NUM', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN'))), ('SELECT_NUM', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('ARITHMETIC_difference', ('SELECT_NUM', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN'))), ('SELECT_NUM', ('PROJECT', 'GET_QUESTION_SPAN', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('ARITHMETIC_difference', ('SELECT_NUM', ('SELECT', 'GET_QUESTION_SPAN')), ('ARITHMETIC_sum', ('SELECT_NUM', ('SELECT', 'GET_QUESTION_SPAN')), ('SELECT_NUM', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('ARITHMETIC_difference', ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')), ('ARITHMETIC_sum', ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')), ('AGGREGATE_count', ('SELECT', 'GET_QUESTION_SPAN')))),
    ('ARITHMETIC_difference', ('AGGREGATE_count', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN')), ('AGGREGATE_count', ('FILTER', ('SELECT', 'GET_QUESTION_SPAN'), 'GET_QUESTION_SPAN'))),
]


""" Split the original Break-annotation into train/dev/test. The original data only comes with train & dev """



def sorted_dict(d: Dict, sort_by_value=True, decreasing=True):
    index = 1 if sort_by_value else 0
    sorted_d = sorted(d.items(), key=lambda x:x[index], reverse=decreasing)
    return sorted_d


def print_verbose_stats(qdmr_examples):
    print("Total QDMR examples: {}".format(len(qdmr_examples)))
    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)

    sorted_templatecount = sorted_dict(template2count)  # [(template, count)] in decreasing count
    print("Number of templates: {}".format(len(template2count)))
    print("Template counts")
    print(" ".join([str(x) for _, x in sorted_templatecount]))


def print_template_complexity(complexity2templates, template2count):
    # Complexity visualization
    sorted_complexity = sorted_dict(complexity2templates, sort_by_value=False, decreasing=False)
    print("Template complexity | count")
    output_str = ""
    for complexity, template_set in sorted_complexity:
        output_str += f"Complexity: {complexity}  Num_templates: {len(template_set)}" + "\n"
        for t in template_set:
            output_str += f"{str(t)}      {template2count[t]}\n"
        output_str += "\n"
    print(output_str)


def downsample_examples(qdmr_examples,
                        lower_limit=20,
                        sample_lower_limit=20, sample_upper_limit=50) -> List[QDMRExample]:
    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)

    # For templates with >20 questions, sample between 20-40 questions only
    selected_qids = []
    for template, qids in template2qids.items():
        if len(qids) < lower_limit:
            selected_qids.extend(qids)
        else:
            num_samples = random.randint(sample_lower_limit, sample_upper_limit)
            random.shuffle(qids)
            selected_qids.extend(qids[:num_samples])

    selected_qdmr_examples = [qid2qdmrexample[qid] for qid in selected_qids]
    return selected_qdmr_examples


def remove_examples_w_infrequent_templates(qdmr_examples, template_count_threshold: int = 1):
    # Removing examples with templates under a certain count threshold
    print("\nRemoving templates with count <= {}".format(template_count_threshold))
    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)
    print("Num of templates: {}".format(len(template2qids)))
    templates = [t for t, c in template2count.items() if c > template_count_threshold]
    print("Remaining templates: {}".format(len(templates)))

    qdmr_examples = [qid2qdmrexample[qid] for t in templates for qid in template2qids[t]]
    return qdmr_examples


def funcs_in_templates(templates: List[str], template2functions: Dict[str, Set[str]]):
    func_set = set()
    for template in templates:
        func_set.update(template2functions[template])
    return func_set


def select_test_templates_manual(template2count, template2funcs):
    train_templates = [t for t in template2count if t not in manual_test_templates]
    test_templates = [t for t in template2count if t in manual_test_templates]

    train_funcs = set([func for t in train_templates for func in template2funcs[t]])
    test_funcs = set([func for t in test_templates for func in template2funcs[t]])

    assert len(test_funcs.difference(train_funcs)) == 0, "Manual split results in unique test functions"

    return train_templates, test_templates


def split_train_dev(tr_qdmrs, dev_ratio):
    random.shuffle(tr_qdmrs)
    num_tr_qdmrs = len(tr_qdmrs)
    std_dev_qdmrs = tr_qdmrs[0:int(dev_ratio * num_tr_qdmrs)]
    std_train_qdmrs = tr_qdmrs[int(dev_ratio * num_tr_qdmrs):]
    return std_train_qdmrs, std_dev_qdmrs


class DownsampleStrategy(Enum):
    full = 0
    only_train = 1


drop_language = DROPLanguage()
def is_drop_parsable(example: QDMRExample) -> bool:
    if not example.drop_nested_expression:
        return False
    program_tree = nested_expression_to_tree(example.drop_nested_expression, predicates_with_strings=True)
    logical_form = nested_expression_to_lisp(program_tree.get_nested_expression())
    try:
        gold_action_sequence: List[str] = drop_language.logical_form_to_action_sequence(logical_form)
        return True
    except:
        return False


def unbiased_test_selection(qdmr_examples: List[QDMRExample], testsize: int, template_limit: int = 20):
    num_examples = len(qdmr_examples)
    print("num of examples: {}".format(num_examples))

    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)

    # List of (template, count)
    sorted_templatecount = sorted_dict(template2count)  # [(template, count)] in decreasing count
    template_counts = [y for _, y in sorted_templatecount]
    print("Number of templates: {}".format(len(template2count)))
    print("Template counts")
    print(" ".join([str(x) for x in template_counts]))

    print(f"Limiting frequent template counts to : {template_limit}.\n"
          f"This is the template distribution we will sample from ... ")
    template_counts = [min(c, template_limit) for c in template_counts]
    print(template_counts)

    num_templates = len(template_counts)
    test_qids = []
    print("sampling test templates w/o replacement ... ")
    while len(test_qids) < testsize:
        counts = np.array(template_counts) - 3   # sampling templates w/ cnt>3 so that there are some left for tr/dev
        counts = np.clip(counts, 0, 100000)
        template_distribution = counts / np.sum(counts)
        template_idx = choice(list(range(num_templates)), 1, p=template_distribution)
        template_idx = int(template_idx)
        sampled_template = sorted_templatecount[template_idx][0]        # This is the chosen template
        sampled_qid = random.choice(template2qids[sampled_template])
        if sampled_qid not in test_qids:
            test_qids.append(sampled_qid)
            template_counts[template_idx] -= 1

    test_examples = [qid2qdmrexample[qid] for qid in test_qids]
    remaining_examples = [example for example in qdmr_examples if example.query_id not in test_qids]
    print("Unbiased test examples stats:")
    print_verbose_stats(test_examples)

    print("\nremaining examples stats:")
    print_verbose_stats(remaining_examples)

    return test_examples, remaining_examples


def data_splits(qdmr_examples: List[QDMRExample], dev_train_ratio: float = 0.1, indomain_unbiased_testsize: int = 500,
                indomain_skewed_testsize: int = 500, unbiased_template_limit: int = 20,
                downsample: bool = False, ds_template_limit=50):

    print("Total QDMR examples: {}".format(len(qdmr_examples)))
    qdmr_examples = [example for example in qdmr_examples if is_drop_parsable(example)]  # examples with programs
    print("Total QDMR examples with programs: {}".format(len(qdmr_examples)))

    qdmr_examples = remove_examples_w_infrequent_templates(qdmr_examples, template_count_threshold=1)
    print("Number of examples: {}".format(len(qdmr_examples)))
    print_verbose_stats(qdmr_examples)
    print("------------------------------------")

    ######################
    print("\nMaking held-out templates test-set")
    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)
    heldout_test_templates = [t for t in template2count if t in manual_test_templates]
    remaining_templates = [t for t in template2count if t not in manual_test_templates]
    print(f"Number of held-out templates: {len(heldout_test_templates)}  Remaining: {len(remaining_templates)}")


    heldout_test_templates = list(heldout_test_templates)
    heldout_test_func_set: Set[str] = funcs_in_templates(templates=heldout_test_templates,
                                                         template2functions=template2funcs)
    heldout_test_examples = [qid2qdmrexample[qid] for t in heldout_test_templates for qid in template2qids[t]]
    print("Number of held-out test questions: {}".format(len(heldout_test_examples)))
    print("Number of test templates: {}".format(len(heldout_test_templates)))
    print("Number of test functions: {}".format(len(heldout_test_func_set)))
    print("held-out example stats:")
    print_verbose_stats(heldout_test_examples)

    remaining_templates = list(remaining_templates)
    rem_func_set: Set[str] = funcs_in_templates(templates=remaining_templates, template2functions=template2funcs)
    remaining_examples = [qid2qdmrexample[qid] for t in remaining_templates for qid in template2qids[t]]
    print("\nNumber of remaining questions: {}".format(len(remaining_examples)))
    print("Number of train templates: {}".format(len(remaining_templates)))
    print("Number of total functions in train: {}".format(len(rem_func_set)))
    print("------------------------------------")

    ######################
    print("\nSelecting in-domain skewed test set of size: {}".format(indomain_skewed_testsize))
    random.shuffle(remaining_examples)
    indomain_skewed_test_examples = remaining_examples[0:indomain_skewed_testsize]
    remaining_examples = remaining_examples[indomain_skewed_testsize:]
    print("Skewed in-domain test set size: {}".format(len(indomain_skewed_test_examples)))
    print("Number of remaining questions: {}".format(len(remaining_examples)))
    print("------------------------------------")

    ######################
    print("\nSelecting in-domain unbiased test set of size: {}".format(indomain_unbiased_testsize))
    indomain_unbiased_test_examples, remaining_examples = unbiased_test_selection(
        qdmr_examples=remaining_examples, testsize=indomain_unbiased_testsize, template_limit=unbiased_template_limit)
    print("remaining examples: {}".format(len(remaining_examples)))
    print("------------------------------------")

    ######################
    print("\nMaking training/dev set from remaining examples")
    if downsample:
        print("Downsampling remaining examples to limit: {}".format(ds_template_limit))
        remaining_examples = downsample_examples(remaining_examples,
                                                 lower_limit=ds_template_limit,
                                                 sample_lower_limit=ds_template_limit,
                                                 sample_upper_limit=ds_template_limit)
        print("After downsampling - ")
        print("Number of examples after downsampling: {}".format(len(remaining_examples)))

    train_examples, dev_examples = split_train_dev(remaining_examples, dev_ratio=dev_train_ratio)
    print("Training data size: {}  Dev data size: {}".format(len(train_examples), len(dev_examples)))

    print("training stats: ")
    print_verbose_stats(train_examples)

    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)
    training_templates = list(template2count.keys())
    tr_func_set: Set[str] = funcs_in_templates(templates=training_templates, template2functions=template2funcs)

    heldout_funcs_not_in_train = heldout_test_func_set.difference(tr_func_set)
    print("Held-out test-set functions not in train: {}".format(heldout_funcs_not_in_train))

    print("------------------------------------")

    return (train_examples, dev_examples, heldout_test_examples,
            indomain_unbiased_test_examples, indomain_skewed_test_examples)


def write_train_dev_test(output_root, train_qdmrs, dev_qdmrs, test_qdmrs):
    print("Writing output to: {}".format(output_root))
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    utils.write_qdmr_examples_to_json(qdmr_examples=train_qdmrs,
                                      qdmr_json=os.path.join(output_root, "train.json"))

    utils.write_qdmr_examples_to_json(qdmr_examples=dev_qdmrs,
                                      qdmr_json=os.path.join(output_root, "dev.json"))

    utils.write_qdmr_examples_to_json(qdmr_examples=test_qdmrs,
                                      qdmr_json=os.path.join(output_root, "test.json"))


def main(args):
    train_qdmr_json = os.path.join(args.drop_dir, "train.json")
    dev_qdmr_json = os.path.join(args.drop_dir, "dev.json")

    train_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(train_qdmr_json)
    dev_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(dev_qdmr_json)

    downsample = args.downsample
    ds_template_limit = args.ds_template_limit

    print("\n\n#######################")
    print("Data splits ")
    all_examples = train_qdmr_examples + dev_qdmr_examples
    (train_examples, dev_examples, heldout_test_examples,
     indomain_unbiased_test_examples, indomain_skewed_test_examples) = data_splits(qdmr_examples=all_examples,
                                                                                   dev_train_ratio=0.15,
                                                                                   indomain_unbiased_testsize=500,
                                                                                   indomain_skewed_testsize=500,
                                                                                   downsample=args.downsample,
                                                                                   ds_template_limit=ds_template_limit)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n\nWriting data to : {output_dir}")
    utils.write_qdmr_examples_to_json(qdmr_examples=train_examples,
                                      qdmr_json=os.path.join(output_dir, "train.json"))

    utils.write_qdmr_examples_to_json(qdmr_examples=dev_examples,
                                      qdmr_json=os.path.join(output_dir, "dev.json"))

    utils.write_qdmr_examples_to_json(qdmr_examples=heldout_test_examples,
                                      qdmr_json=os.path.join(output_dir, "heldout_test.json"))

    utils.write_qdmr_examples_to_json(qdmr_examples=indomain_unbiased_test_examples,
                                      qdmr_json=os.path.join(output_dir, "indomain_unbiased_test.json"))

    utils.write_qdmr_examples_to_json(qdmr_examples=indomain_skewed_test_examples,
                                      qdmr_json=os.path.join(output_dir, "indomain_skewed_test.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument('--downsample', dest='downsample', action='store_true')
    parser.add_argument("--ds_template_limit", type=int, default=50)


    args = parser.parse_args()

    main(args)
