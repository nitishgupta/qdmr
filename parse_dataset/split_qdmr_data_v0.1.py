from typing import List, Dict, Set, Tuple

import os
import random
import argparse
from collections import defaultdict

from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, convert_nestedexpr_to_tuple, \
    get_inorder_function_list_from_template

from qdmr.data import utils
from analysis.qdmr_program_diversity import get_maps
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


def select_test_tempalates(probable_template_list: List[Tuple], template2count, target_test_questions: int,
                           train_templates: List[Tuple], template2funcs: Dict[Tuple, Set[str]], downsample: bool,
                           common_template_thresholds: List[int]):
    test_templates_options: List[Set] = []
    test_numques_options: List[int] = []
    train_templates_options: List[Set] = []

    print("Target number of test questions: {}".format(target_test_questions))

    # print(len(probable_template_set))
    # new_probable_set = remove_templates_w_unique_function(probable_template_set, template2funcs)
    # print(len(new_probable_set))
    # exit()

    num_chances = 20000
    for chance in range(num_chances):
        if chance % 2000 == 0:
            print("tries: {}".format(chance))
        # template_list = copy.deepcopy(probable_template_list)

        template_idxs = list(range(len(probable_template_list)))
        random.shuffle(template_idxs)
        test_templates = set()
        template_num = 0
        num_test_ques = 0
        for template_num in template_idxs:
            if num_test_ques >= target_test_questions:
                break
            template = probable_template_list[template_num]
            thrshold = max(common_template_thresholds) if not downsample else min(common_template_thresholds)
            if template2count[template] > thrshold:     # don't choose frequent templates for testing
                continue
            test_templates.add(template)
            num_test_ques += template2count[template]

        full_train_templates = set(train_templates)
        full_train_templates.update([t for t in probable_template_list if t not in test_templates])
        train_funcs = set([func for t in full_train_templates for func in template2funcs[t]])
        test_funcs = set([func for t in test_templates for func in template2funcs[t]])

        testfuncs_diff_train = test_funcs.difference(train_funcs)
        if len(testfuncs_diff_train) == 0:  # all functions in test are seen in train
            test_templates_options.append(test_templates)
            train_templates_options.append(full_train_templates)
            test_numques_options.append(num_test_ques)

    print("Number of options : {}".format(len(test_templates_options)))
    index, t_templates = max(enumerate(test_templates_options), key=lambda x: len(x[1]))
    test_templates = t_templates
    # num_t_temps = len(t_templates)
    # all_templates_options = [temps for temps in test_templates_options if len(temps) == num_t_temps]
    # num_ques_in_options = [sum([template2count[t] for t in test_templates]) for test_templates in all_templates_options]
    # index_templates_w_max_ques = max(enumerate(num_ques_in_options), key=lambda x: x[1])
    # test_templates = all_templates_options[index_templates_w_max_ques[0]]
    test_num_questions = sum([template2count[t] for t in test_templates])
    print("Number of test questions: {}".format(test_num_questions))
    print("Number of test templates: {}".format(len(test_templates)))
    for template in test_templates:
        print(f"{template}    {template2count[template]}")

    full_train_templates = set(train_templates)
    full_train_templates.update([t for t in probable_template_list if t not in test_templates])

    return full_train_templates, test_templates


def standard_split(train_examples: List[QDMRExample], test_examples: List[QDMRExample], train_dev_ratio: float = 0.1,
                   downsample: bool = False):

    train_qids = [example.query_id for example in train_examples]
    test_qids = [example.query_id for example in test_examples]

    print("Number of train examples: {}".format(len(train_examples)))
    print("Number of test examples: {}".format(len(test_examples)))

    all_examples = remove_examples_w_infrequent_templates(train_examples + test_examples, template_count_threshold=1)
    train_examples = [example for example in all_examples if example.query_id in train_qids]
    test_examples = [example for example in all_examples if example.query_id in test_qids]

    print("Number of train examples: {}".format(len(train_examples)))
    print("Number of test examples: {}".format(len(test_examples)))

    # Standard-split
    random.shuffle(train_examples)
    std_train_qdmrs = train_examples[int(train_dev_ratio * len(train_examples)):]
    std_dev_qdmrs = train_examples[0:int(train_dev_ratio * len(train_examples))]
    std_test_qdmrs = test_examples
    print("Train: ")
    print_verbose_stats(std_train_qdmrs)
    print("Dev: ")
    print_verbose_stats(std_dev_qdmrs)

    if downsample:
        print("\nDownsampling examples with frequent templates ... ")
        print("Train examples before downsampling: {}".format(len(std_train_qdmrs)))
        std_train_qdmrs = downsample_examples(std_train_qdmrs,
                                              lower_limit=50,
                                              sample_lower_limit=50,
                                              sample_upper_limit=50)
        print("Train examples after downsampling: {}".format(len(std_train_qdmrs)))
        print_verbose_stats(std_train_qdmrs)
        # train_examples = [example for example in all_examples if example.query_id in train_qids]
        # test_examples = [example for example in all_examples if example.query_id in test_qids]
        # print("Number of train examples: {}".format(len(train_examples)))
        # print("Number of test examples: {}".format(len(test_examples)))

    print("\nStandard-split")
    print(f"Train: {len(std_train_qdmrs)}  Dev: {len(std_dev_qdmrs)}  Test: {len(std_test_qdmrs)}")

    return std_train_qdmrs, std_dev_qdmrs, std_test_qdmrs


def template_split(train_qdmr_examples: List[QDMRExample], dev_qdmr_examples: List[QDMRExample],
                   test_ratio: float = 0.15, dev_train_ratio: float = 0.1, downsample: bool = False,
                   manual_test: bool = False):
    """ This data is processed using parse_dataset.parse_qdmr and keys in this json can be glanced at from there.

    The nested_expression in the data makes life easier since functions are already normalized to their min/max, add/sub
    identifier.

    train_ratio: ``float`` Is the ratio of train_ques to total questions available
    """

    # Merge original train and dev to make new splits
    qdmr_examples = train_qdmr_examples + dev_qdmr_examples
    print("Total QDMR examples: {}".format(len(qdmr_examples)))
    qdmr_examples = [example for example in qdmr_examples if example.program_tree]  # examples with programs
    print("Total QDMR examples with programs: {}".format(len(qdmr_examples)))

    # if downsample:
    #     qdmr_examples = downsample_examples(qdmr_examples,
    #                                         lower_limit=50,
    #                                         sample_lower_limit=50,
    #                                         sample_upper_limit=50)
    #     print("Total QDMR examples after downsampling: {}".format(len(qdmr_examples)))

    # print_verbose_stats(qdmr_examples)

    print()
    # Removing examples with templates under a certain count threshold
    qdmr_examples = remove_examples_w_infrequent_templates(qdmr_examples, template_count_threshold=1)
    print_verbose_stats(qdmr_examples)

    # Train / Test template split
    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)

    if manual_test:
        print("\nPerforming manual template split ... ")
        train_templates = [t for t in template2count if t not in manual_test_templates]
        test_templates = [t for t in template2count if t in manual_test_templates]
    else:
        print("\nPerfomring automatic template split ... ")
        # Test will only contain examples from these complexities
        test_complexities = [5, 7, 8, 9]
        # These are definitely train templates
        train_templates = []
        # Test templates will be carved out from this. remaining would be added to train_templates
        probable_templates = []
        for qdmr_example in qdmr_examples:
            qid = qdmr_example.query_id
            template = qid2template[qid]
            complexity = len(get_inorder_function_list_from_template(template))
            if complexity in test_complexities:
                probable_templates.append(template)
            else:
                train_templates.append(template)

        target_num_test_questions = int(test_ratio * len(qdmr_examples))
        train_templates, test_templates = select_test_tempalates(probable_template_list=probable_templates,
                                                                 template2count=template2count,
                                                                 target_test_questions=target_num_test_questions,
                                                                 train_templates=train_templates,
                                                                 template2funcs=template2funcs,
                                                                 # Setting this true since we're not downsampling before
                                                                 # and want to use the 300 threshold
                                                                 downsample=True,
                                                                 common_template_thresholds=[300, 30])


    print("Number of train templates: {}".format(len(train_templates)))
    print("Number of test templates: {}".format(len(test_templates)))

    ###
    train_templates = list(train_templates)
    num_train_ques = sum(template2count[t] for t in train_templates)
    tr_func_set: Set[str] = funcs_in_templates(templates=train_templates, template2functions=template2funcs)
    print("\nNumber of train questions: {}".format(num_train_ques))
    print("Number of train templates: {}".format(len(train_templates)))
    print("Number of total functions in train: {}".format(len(tr_func_set)))

    # test_complexities = set([template2complexity[t] for t in test_templates])
    # print(test_complexities)
    test_templates = list(test_templates)
    test_func_set: Set[str] = funcs_in_templates(templates=test_templates, template2functions=template2funcs)
    print("Number of test templates: {}".format(len(test_templates)))
    print("Number of test functions: {}".format(len(test_func_set)))

    print("Function not in train: {}".format(test_func_set.difference(tr_func_set)))
    print("Function not in test: {}".format(tr_func_set.difference(test_func_set)))

    train_qdmrs = []
    # In-domain dev set
    dev_in_qdmrs = []
    dev_train_in_ratio = dev_train_ratio
    train_dev_in_ratio = 1.0 - dev_train_in_ratio
    for t in train_templates:
        t_qids = template2qids[t]
        # Split qids for this template into 0.9/0.1 for train/dev-in
        t_train_qids = t_qids[0:int(train_dev_in_ratio * len(t_qids))]
        t_dev_qids = t_qids[int(train_dev_in_ratio * len(t_qids)):]
        train_qdmrs.extend([qid2qdmrexample[qid] for qid in t_train_qids])
        dev_in_qdmrs.extend([qid2qdmrexample[qid] for qid in t_dev_qids])

    print("Train: ")
    print_verbose_stats(train_qdmrs)
    print("Dev: ")
    print_verbose_stats(dev_in_qdmrs)

    if downsample:
        print("\nDownsampling training examples with frequent templates ... ")
        print("Train examples before downsampling: {}".format(len(train_qdmrs)))
        train_qdmrs = downsample_examples(train_qdmrs,
                                          lower_limit=50,
                                          sample_lower_limit=50,
                                          sample_upper_limit=50)
        print("Train examples after downsampling: {}".format(len(train_qdmrs)))
        print_verbose_stats(train_qdmrs)

    test_qdmrs = [qid2qdmrexample[qid] for t in test_templates for qid in template2qids[t]]

    print("\nTemplate-split")
    print(f"Train: {len(train_qdmrs)}  Dev: {len(dev_in_qdmrs)}  Test: {len(test_qdmrs)}")

    """ Deprecated 
    test_qids = []
    test_qdmrs = []
    # Out-of-domain dev set
    dev_out_qids = []
    dev_out_qdmrs = []
    dev_test_out_ratio = 0.00
    for t in test_templates:
        t_qids = template2qids[t]
        # Split qids for this template into 0.1/0.9 for dev-out/test
        t_dev_qids = t_qids[0:int(dev_test_out_ratio * len(t_qids))]
        t_test_qids = t_qids[int(dev_test_out_ratio * len(t_qids)):]
        dev_out_qids.extend(t_dev_qids)
        test_qids.extend(t_test_qids)
        dev_out_qdmrs.extend([qid2qdmrexample[qid] for qid in t_dev_qids])
        test_qdmrs.extend([qid2qdmrexample[qid] for qid in t_test_qids])

    print(f"Number of questions; train:{len(train_qdmrs)} dev-in:{len(dev_in_qdmrs)} "
          f"dev-out:{len(dev_out_qdmrs)} test:{len(test_qdmrs)}")
    """

    return train_qdmrs, dev_in_qdmrs, test_qdmrs


# def template_split(train_qdmr_examples: List[QDMRExample], dev_qdmr_examples: List[QDMRExample],
#                    test_ratio: float = 0.15, dev_train_ratio: float = 0.1, downsample: bool = False,
#                    manual_test: bool = False):
#     """ This data is processed using parse_dataset.parse_qdmr and keys in this json can be glanced at from there.
#
#     The nested_expression in the data makes life easier since functions are already normalized to their min/max, add/sub
#     identifier.
#
#     train_ratio: ``float`` Is the ratio of train_ques to total questions available
#     """
#
#     # Merge original train and dev to make new splits
#     qdmr_examples = train_qdmr_examples + dev_qdmr_examples
#     print("Total QDMR examples: {}".format(len(qdmr_examples)))
#     qdmr_examples = [example for example in qdmr_examples if example.program_tree]  # examples with programs
#     print("Total QDMR examples with programs: {}".format(len(qdmr_examples)))
#
#     if downsample:
#         qdmr_examples = downsample_examples(qdmr_examples,
#                                             lower_limit=50,
#                                             sample_lower_limit=50,
#                                             sample_upper_limit=50)
#         print("Total QDMR examples after downsampling: {}".format(len(qdmr_examples)))
#
#     print_verbose_stats(qdmr_examples)
#
#     print()
#     # Removing examples with templates under a certain count threshold
#     template_count_threshold = 1
#     print("Removing templates with count <= {}".format(template_count_threshold))
#     qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
#         get_maps(qdmr_examples)
#     templates = [t for t, c in template2count.items() if c > template_count_threshold]
#     print("Remaining templates: {}".format(len(templates)))
#
#     qdmr_examples = [qid2qdmrexample[qid] for t in templates for qid in template2qids[t]]
#     print_verbose_stats(qdmr_examples)
#
#     # Train / Test template split
#     qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
#         get_maps(qdmr_examples)
#
#     # print_template_complexity(complexity2templates, template2count)
#
#     if manual_test:
#         print("\nPerfomring manual template split ... ")
#         train_templates = [t for t in template2count if t not in manual_test_templates]
#         test_templates = [t for t in template2count if t in manual_test_templates]
#     else:
#         print("\nPerfomring automatic template split ... ")
#         # Test will only contain examples from these complexities
#         test_complexities = [5, 7, 8, 9]
#         # These are definitely train templates
#         train_templates = []
#         # Test templates will be carved out from this. remaining would be added to train_templates
#         probable_templates = []
#         for qdmr_example in qdmr_examples:
#             qid = qdmr_example.query_id
#             template = qid2template[qid]
#             complexity = len(get_inorder_function_list_from_template(template))
#             if complexity in test_complexities:
#                 probable_templates.append(template)
#             else:
#                 train_templates.append(template)
#
#         target_num_test_questions = int(test_ratio * len(qdmr_examples))
#         train_templates, test_templates = select_test_tempalates(probable_template_list=probable_templates,
#                                                                  template2count=template2count,
#                                                                  target_test_questions=target_num_test_questions,
#                                                                  train_templates=train_templates,
#                                                                  template2funcs=template2funcs,
#                                                                  downsample=downsample,
#                                                                  common_template_thresholds=[300, 30])
#
#
#     print("Number of train templates: {}".format(len(train_templates)))
#     print("Number of test templates: {}".format(len(test_templates)))
#
#     ###
#     train_templates = list(train_templates)
#     num_train_ques = sum(template2count[t] for t in train_templates)
#     tr_func_set: Set[str] = funcs_in_templates(templates=train_templates, template2functions=template2funcs)
#     print("\nNumber of train questions: {}".format(num_train_ques))
#     print("Number of train templates: {}".format(len(train_templates)))
#     print("Number of total functions in train: {}".format(len(tr_func_set)))
#
#     # test_complexities = set([template2complexity[t] for t in test_templates])
#     # print(test_complexities)
#     test_templates = list(test_templates)
#     test_func_set: Set[str] = funcs_in_templates(templates=test_templates, template2functions=template2funcs)
#     print("Number of test templates: {}".format(len(test_templates)))
#     print("Number of test functions: {}".format(len(test_func_set)))
#
#     print("Function not in train: {}".format(test_func_set.difference(tr_func_set)))
#     print("Function not in test: {}".format(tr_func_set.difference(test_func_set)))
#
#     train_qids = []
#     train_qdmrs = []
#     # In-domain dev set
#     dev_in_qids = []
#     dev_in_qdmrs = []
#     dev_train_in_ratio = dev_train_ratio
#     for t in train_templates:
#         t_qids = template2qids[t]
#         # Split qids for this template into 0.9/0.1 for train/dev-in
#         t_dev_qids = t_qids[0:int(dev_train_in_ratio * len(t_qids))]
#         t_train_qids = t_qids[int(dev_train_in_ratio * len(t_qids)):]
#         train_qids.extend(t_train_qids)
#         dev_in_qids.extend(t_dev_qids)
#         train_qdmrs.extend([qid2qdmrexample[qid] for qid in t_train_qids])
#         dev_in_qdmrs.extend([qid2qdmrexample[qid] for qid in t_dev_qids])
#
#     test_qdmrs = [qid2qdmrexample[qid] for t in test_templates for qid in template2qids[t]]
#
#     print("\nTemplate-split")
#     print(f"Train: {len(train_qdmrs)}  Dev: {len(dev_in_qdmrs)}  Test: {len(test_qdmrs)}")
#
#     """ Deprecated
#     test_qids = []
#     test_qdmrs = []
#     # Out-of-domain dev set
#     dev_out_qids = []
#     dev_out_qdmrs = []
#     dev_test_out_ratio = 0.00
#     for t in test_templates:
#         t_qids = template2qids[t]
#         # Split qids for this template into 0.1/0.9 for dev-out/test
#         t_dev_qids = t_qids[0:int(dev_test_out_ratio * len(t_qids))]
#         t_test_qids = t_qids[int(dev_test_out_ratio * len(t_qids)):]
#         dev_out_qids.extend(t_dev_qids)
#         test_qids.extend(t_test_qids)
#         dev_out_qdmrs.extend([qid2qdmrexample[qid] for qid in t_dev_qids])
#         test_qdmrs.extend([qid2qdmrexample[qid] for qid in t_test_qids])
#
#     print(f"Number of questions; train:{len(train_qdmrs)} dev-in:{len(dev_in_qdmrs)} "
#           f"dev-out:{len(dev_out_qdmrs)} test:{len(test_qdmrs)}")
#     """
#
#     return train_qdmrs, dev_in_qdmrs, test_qdmrs


def main(args):
    train_qdmr_json = args.train_qdmr_json
    dev_qdmr_json = args.dev_qdmr_json

    output_dir = args.out_dir

    train_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(train_qdmr_json)
    dev_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(dev_qdmr_json)

    if args.split in ["standard", "std"]:
        train_qdmrs, dev_qdmrs, test_qdmrs = standard_split(train_examples=train_qdmr_examples,
                                                            test_examples=dev_qdmr_examples,
                                                            train_dev_ratio=0.1,
                                                            downsample=args.downsample)
    elif args.split in ["template", "temp", "tmp"]:
        train_qdmrs, dev_qdmrs, test_qdmrs = template_split(train_qdmr_examples=train_qdmr_examples,
                                                            dev_qdmr_examples=dev_qdmr_examples,
                                                            test_ratio=0.15,
                                                            dev_train_ratio=0.1,
                                                            downsample=args.downsample,
                                                            manual_test=args.manual_test)
    else:
        print("Incorrect split: {}".format(args.split))
        raise ValueError

    print("Writing output to: {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    utils.write_qdmr_examples_to_json(qdmr_examples=train_qdmrs,
                                      qdmr_json=os.path.join(output_dir, "train.json"))

    utils.write_qdmr_examples_to_json(qdmr_examples=dev_qdmrs,
                                      qdmr_json=os.path.join(output_dir, "dev.json"))

    utils.write_qdmr_examples_to_json(qdmr_examples=test_qdmrs,
                                      qdmr_json=os.path.join(output_dir, "test.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_qdmr_json", required=True)
    parser.add_argument("--dev_qdmr_json", required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument('--downsample', dest='downsample', action='store_true')
    parser.add_argument('--manual_test', dest='manual_test', action='store_true')

    args = parser.parse_args()

    if args.split not in ["standard", "template", "std", "temp", "tmp"]:
        print("Split not defined")
        raise ValueError

    main(args)
