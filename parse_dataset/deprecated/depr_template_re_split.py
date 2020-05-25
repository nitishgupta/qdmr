from typing import List, Dict, Set, Tuple

import os
import copy
import random
import argparse
from collections import defaultdict

from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, convert_nestedexpr_to_tuple, \
    get_inorder_function_list, Node, get_inorder_function_list_from_template, write_qdmr_examples_to_json

from analysis.qdmr_program_diversity import get_maps

random.seed(1)


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



def select_test_tempalates(probable_template_list: List[Tuple], template2count, target_test_questions: int,
                           train_templates: List[Tuple], template2funcs: Dict[Tuple, Set[str]], downsample: bool,
                           common_template_thresholds: List[int]):
    test_templates_options: List[Set] = []
    train_templates_options: List[Set] = []

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
            if template2count[template] > thrshold:
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
            # print("Aye")
        # else:
            # print(len(testfuncs_diff_train))
            # print(testfuncs_diff_train)

    print("Number of options : {}".format(len(test_templates_options)))
    mm = max(enumerate(test_templates_options), key=lambda x: len(x[1]))
    print(f"index: {mm[0]}  num_templates: {len(mm[1])}")
    test_templates = test_templates_options[mm[0]]
    test_num_questions = sum([template2count[t] for t in test_templates])
    print("Number of test questions: {}".format(test_num_questions))
    print("Number of test templates: {}".format(len(test_templates)))
    for template in test_templates:
        print(f"{template}    {template2count[template]}")

    full_train_templates = set(train_templates)
    full_train_templates.update([t for t in probable_template_list if t not in test_templates])

    return full_train_templates, test_templates


def sorted_dict(d: Dict, sort_by_value=True, decreasing=True):
    index = 1 if sort_by_value else 0
    sorted_d = sorted(d.items(), key=lambda x:x[index], reverse=decreasing)
    return sorted_d


def funcs_in_templates(templates: List[str], template2functions: Dict[str, Set[str]]):
    func_set = set()
    for template in templates:
        func_set.update(template2functions[template])
    return func_set



# def remove_templates_w_unique_function(templates, template2functions: Dict[Tuple, Set[str]]):
#     func2templates = defaultdict(set)
#     for t in templates:
#         functions = template2functions[t]
#         for func in functions:
#             func2templates[func].add(t)
#
#     filtered_templates = set()
#     for f, temps in func2templates.items():
#         if len(temps) > 1:
#             filtered_templates.update(temps)
#
#     return list(filtered_templates)



def print_verbose_stats(qdmr_examples):
    print("Total QDMR examples: {}".format(len(qdmr_examples)))
    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)

    sorted_templatecount = sorted_dict(template2count)  # [(template, count)] in decreasing count
    print("Number of templates: {}".format(len(template2count)))
    print("Template counts")
    print(" ".join([str(x) for _, x in sorted_templatecount]))

    """
    # Complexity visualization
    sorted_complexity = sorted_dict(complexity2templates, sort_by_value=False, decreasing=False)
    print("Template complexity | count")
    output_str = ""
    for complexity, template_set in sorted_complexity:
        output_str += f"Complexity: {complexity}  Num_templates: {len(template_set)}" + "\n"
        # print("\n".join([str(t) for t in template_set]))   # To print all templates
    print(output_str)
    print()
    """




def template_based_split(train_qdmr_examples: List[QDMRExample], dev_qdmr_examples: List[QDMRExample],
                         train_ratio: float, downsample: bool):
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

    if downsample:
        qdmr_examples = downsample_examples(qdmr_examples, sample_upper_limit=50)
        print("Total QDMR examples after downsampling: {}".format(len(qdmr_examples)))

    print_verbose_stats(qdmr_examples)

    print()
    # Removing examples with templates under a certain count threshold
    template_count_threshold = 1
    print("Removing templates with count <= {}".format(template_count_threshold))
    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)
    templates = [t for t, c in template2count.items() if c > template_count_threshold]
    print("Remaining templates: {}".format(len(templates)))

    qdmr_examples = [qid2qdmrexample[qid] for t in templates for qid in template2qids[t]]
    print_verbose_stats(qdmr_examples)

    # Train / Test template split
    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)

    # Test will only contain examples from these complexities
    test_complexities = [5, 7, 9]
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

    target_num_test_questions = int(0.1*len(qdmr_examples))
    train_templates, test_templates = select_test_tempalates(probable_template_list=probable_templates,
                                                             template2count=template2count,
                                                             target_test_questions=target_num_test_questions,
                                                             train_templates=train_templates,
                                                             template2funcs=template2funcs,
                                                             downsample=downsample,
                                                             common_template_thresholds=[300, 30])


    print("Number of train templates: {}".format(len(train_templates)))
    print("Number of test templates: {}".format(len(test_templates)))

    ###
    train_templates = list(train_templates)
    num_train_ques = sum(template2count[t] for t in train_templates)
    tr_func_set: Set[str] = funcs_in_templates(templates=train_templates, template2functions=template2funcs)
    print("Number of train questions: {}".format(num_train_ques))
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

    train_qids = []
    train_qdmrs = []
    # In-domain dev set
    dev_in_qids = []
    dev_in_qdmrs = []
    dev_train_in_ratio = 0.1
    for t in train_templates:
        t_qids = template2qids[t]
        # Split qids for this template into 0.9/0.1 for train/dev-in
        t_dev_qids = t_qids[0:int(dev_train_in_ratio * len(t_qids))]
        t_train_qids = t_qids[int(dev_train_in_ratio * len(t_qids)):]
        train_qids.extend(t_train_qids)
        dev_in_qids.extend(t_dev_qids)
        train_qdmrs.extend([qid2qdmrexample[qid] for qid in t_train_qids])
        dev_in_qdmrs.extend([qid2qdmrexample[qid] for qid in t_dev_qids])

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

    return train_qdmrs, dev_in_qdmrs, dev_out_qdmrs, test_qdmrs


def main(args):
    train_qdmr_json = args.train_qdmr_json
    dev_qdmr_json = args.dev_qdmr_json

    train_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(train_qdmr_json)
    dev_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(dev_qdmr_json)

    (tmp_based_train_qdmrs,
     tmp_based_dev_in_qdmrs,
     tmp_based_dev_out_qdmrs,
     tmp_based_test_qdmrs) = template_based_split(train_qdmr_examples, dev_qdmr_examples, train_ratio=args.train_ratio,
                                                  downsample=args.downsample)

    # In the args.tmp_split_dir directory write 4 files -- train.json, dev-in.json, dev-out.json, test.json
    print("Writing output to: {}".format(args.tmp_split_dir))
    if not os.path.exists(args.tmp_split_dir):
        os.makedirs(args.tmp_split_dir, exist_ok=True)

    write_qdmr_examples_to_json(qdmr_examples=tmp_based_train_qdmrs,
                                qdmr_json=os.path.join(args.tmp_split_dir, "train.json"))

    write_qdmr_examples_to_json(qdmr_examples=tmp_based_dev_in_qdmrs,
                                qdmr_json=os.path.join(args.tmp_split_dir, "dev.json"))

    if tmp_based_dev_out_qdmrs:
        write_qdmr_examples_to_json(qdmr_examples=tmp_based_dev_out_qdmrs,
                                    qdmr_json=os.path.join(args.tmp_split_dir, "dev-out.json"))

    write_qdmr_examples_to_json(qdmr_examples=tmp_based_test_qdmrs,
                                qdmr_json=os.path.join(args.tmp_split_dir, "test.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_qdmr_json", required=True)
    parser.add_argument("--dev_qdmr_json", required=True)
    parser.add_argument("--tmp_split_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    # parser.add_argument("--downsample", type=bool, default=True)
    parser.add_argument('--downsample', dest='downsample', action='store_true')

    args = parser.parse_args()

    main(args)
