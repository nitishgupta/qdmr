from typing import List, Dict
import json
from collections import defaultdict
import random
import argparse

from qdmr.utils import read_qdmr_json_to_examples, QDMRExample, Node, nested_expression_to_tree, \
    nested_expression_to_lisp, string_arg_to_quesspan_pred

from qdmr import utils

random.seed(28)


def get_operators(nested_expression):
    function_names = set()
    operator_template = []
    for i, argument in enumerate(nested_expression):
        if i == 0:
            function_names.add(argument)
            operator_template.append(argument)
        else:
            if isinstance(argument, list):
                func_set, op_template = get_operators(argument)
                function_names.update(func_set)
                operator_template.extend(op_template)

    return function_names, operator_template


def read_qdmr(qdmr_examples: List[QDMRExample]):
    """ This data is processed using parse_dataset.parse_qdmr and keys in this json can be glanced at from there.

    The nested_expression in the data makes life easier since functions are already normalized to their min/max, add/sub
    identifier.

    This function mainly reads relevant

    :param qdmr_json:
    :return:
    """
    total_ques = 0
    total_super = 0

    qid2ques = {}
    qid2program = {}
    function2qids = defaultdict(list)
    operatortemplate2count = defaultdict(int)
    qid2nestedexp = {}
    for qdmr_example in qdmr_examples:
        query_id = qdmr_example.query_id
        question = qdmr_example.question
        program: List[str] = qdmr_example.program
        nested_expression: List = qdmr_example.nested_expression
        typed_nested_expression: List = qdmr_example.typed_nested_expression

        program_tree = nested_expression_to_tree(typed_nested_expression)
        program_tree = string_arg_to_quesspan_pred(node=program_tree)


        qid2ques[query_id] = question
        qid2program[query_id] = program
        qid2nestedexp[query_id] = nested_expression

        # Skip questions w/o program supervision
        if "None" in program:
            continue

        functions_set, operator_template = get_operators(nested_expression)

        for func in functions_set:
            function2qids[func].append(query_id)

        operator_template = tuple(operator_template)

        operatortemplate2count[operator_template] += 1

        # Some operators such as AGGREGATE, COMPARISON, etc. are high-level and need to be distinguished.
        # normalized_operators = []
        # for step in program:
        #     operator = step.split("[")[0]
        #
        #     if operator in ["AGGREGATE", "COMPARISON", "GROUP", "ARITHMETIC", "SUPERLATIVE"]:
        #         func = step.split("[")[1].split(",")[0][1:-1]
        #         norm_operator = operator + "_" + func
        #     else:
        #         norm_operator = operator
        #     normalized_operators.append(norm_operator)
        # normalized_operators = tuple(normalized_operators)

        total_ques += 1

    print("Total questions: {}  Total program abstractions: {}".format(total_ques, len(operatortemplate2count)))
    return qid2ques, operatortemplate2count, function2qids, qid2nestedexp


def train_dev_stats(train_optemplate2count, train_func2qids, dev_optemplate2count=None, dev_func2qids=None):
    train_templates = set(train_optemplate2count.keys())
    print("Train number of program templates: {}".format(len(train_templates)))
    print("Train number of unique functions: {}".format(len(train_func2qids)))
    print()

    if dev_optemplate2count is not None:
        dev_templates = set(dev_optemplate2count.keys())
        print("Dev number of program templates: {}".format(len(dev_templates)))
        train_dev_common = train_templates.intersection(dev_templates)
        dev_extra_templates = dev_templates.difference(train_templates)
        print("Train / Dev common program templates (this disregards arguments): {}".format(len(train_dev_common)))
        print("Dev extra abstract program templates: {}".format(len(dev_extra_templates)))
    print()



def get_maps(qdmr_examples: List[QDMRExample]):
    """ This data is processed using parse_dataset.parse_qdmr and keys in this json can be glanced at from there.

    The nested_expression in the data makes life easier since functions are already normalized to their min/max, add/sub
    identifier.

    This function mainly reads relevant
    """

    qid2qdmrexample = {}
    for qdmr_example in qdmr_examples:
        qid2qdmrexample[qdmr_example.query_id] = qdmr_example


    pred2qids = defaultdict(list)
    for qdmr_example in qdmr_examples:
        qid = qdmr_example.query_id
        if not qdmr_example.typed_masked_nested_expr:
            continue
        func_set, template = utils.convert_nestedexpr_to_tuple(qdmr_example.typed_masked_nested_expr)
        for pred in func_set:
            pred2qids[pred].append(qid)

    return qid2qdmrexample, pred2qids


def write_example_programs_tsv(output_tsv_path,
                               qid2qdmrexample: Dict[str, QDMRExample], pred2qids: Dict[str, List[str]]):
    """Write example (question, program) in TSV for Google Sheets.

    To ensure diversity, we first sample 10 questions for each function type.
    """
    print("Writing examples to TSV: {}".format(output_tsv_path))
    qid_ques_programs = []
    for func, qids in pred2qids.items():
        print(func)
        random.shuffle(qids)
        for i in range(0, min(15, len(qids))):
            qid = qids[i]
            qdmr_example = qid2qdmrexample[qid]
            nested_expr = qdmr_example.program_tree._get_nested_expression_with_strings()
            qid_ques_programs.append((func,
                                      qid,
                                      qdmr_example.question,
                                      nested_expr))

    print("Total examples written: {}".format(len(qid_ques_programs)))

    with open(output_tsv_path, 'w') as outf:
        outf.write(f"Function\tQueryID\tQuestion\tProgram\n")
        # random.shuffle(qid_programs)
        for (func, qid, ques, program) in qid_ques_programs:
            out_str = f"{func}\t{qid}\t{ques}\t{program}\n"
            outf.write(out_str)


def main(args):
    train_qdmr_json = args.train_qdmr_json
    dev_qdmr_json = args.dev_qdmr_json

    output_tsv_path = args.output_tsv_path

    train_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(train_qdmr_json)
    dev_qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(dev_qdmr_json)

    qid2qdmrexample, pred2qids = get_maps(train_qdmr_examples)

    write_example_programs_tsv(output_tsv_path, qid2qdmrexample, pred2qids)

    #
    #
    # train_qid2ques, train_optemplate2count, train_func2qids, train_qid2nestedexp = read_qdmr(train_qdmr_json)
    # dev_qid2ques, dev_optemplate2count, dev_func2qids, dev_qid2nestedexp = None, None, None, None
    # if dev_qdmr_json is not None:
    #     dev_qid2ques, dev_optemplate2count, dev_func2qids, dev_qid2nestedexp = read_qdmr(dev_qdmr_json)
    #
    # train_dev_stats(train_optemplate2count, train_func2qids,dev_optemplate2count, dev_func2qids)
    #
    # if output_tsv_path:
    #     write_example_programs_tsv(output_tsv_path, train_qid2ques, train_qid2nestedexp, train_func2qids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_qdmr_json")
    parser.add_argument("--dev_qdmr_json")
    parser.add_argument("--output_tsv_path")
    args = parser.parse_args()

    main(args)
