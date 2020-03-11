from typing import List, Dict
import json
from collections import defaultdict
import random
import argparse

from qdmr.utils import read_qdmr_json_to_examples, QDMRExample, Node, nested_expression_to_tree, \
    nested_expression_to_lisp, string_arg_to_quesspan_pred

random.seed(28)

train_qdmr_json = " /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/train.json"
dev_qdmr_json = " /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/dev.json "


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



def convert_nestedexpr_to_tuple(nested_expression):
    new_nested = []
    for i, argument in enumerate(nested_expression):
        if i == 0:
            new_nested.append(argument)
        else:
            if isinstance(argument, list):
                tupled_nested = convert_nestedexpr_to_tuple(argument)
                new_nested.append(tupled_nested)
            else:
                new_nested.append(argument)
    return tuple(new_nested)



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
    qid2optemplate = {}
    function2qids = defaultdict(list)
    operatortemplate2count = defaultdict(int)
    operatortemplate2qids = defaultdict(list)
    qid2nestedexp = {}

    for qdmr_example in qdmr_examples:
        query_id = qdmr_example.query_id
        question = qdmr_example.question
        program: List[str] = qdmr_example.program
        nested_expression: List = qdmr_example.nested_expression
        typed_nested_expression: List = qdmr_example.typed_nested_expression

        # Skip examples without typed program
        if not len(typed_nested_expression):
            continue

        program_tree = nested_expression_to_tree(typed_nested_expression)
        program_tree = string_arg_to_quesspan_pred(node=program_tree)
        masked_nested_expr = program_tree.get_nested_expression()

        # function_names, operator_template = get_operators(masked_nested_expr)
        operator_template = convert_nestedexpr_to_tuple(masked_nested_expr) # tuple(operator_template)

        qid2ques[query_id] = question
        qid2optemplate[query_id] = operator_template
        qid2nestedexp[query_id] = program_tree._get_nested_expression_with_strings()
        operatortemplate2count[operator_template] += 1
        operatortemplate2qids[operator_template].append(query_id)

        total_ques += 1

    print("Total questions: {}  Total program abstractions: {}".format(total_ques, len(operatortemplate2count)))
    return qid2ques, operatortemplate2count, operatortemplate2qids, qid2nestedexp


def train_dev_stats(train_qid2ques, train_optemplate2count, train_optemplate2qids,
                    dev_qid2ques, dev_optemplate2count=None, dev_optemplate2qids=None):
    train_templates = set(train_optemplate2count.keys())
    print("Train number of program templates: {}".format(len(train_templates)))
    # print("Train number of unique functions: {}".format(len(train_func2qids)))
    print()



    if dev_optemplate2count is not None:
        dev_templates = set(dev_optemplate2count.keys())
        print("Dev number of program templates: {}".format(len(dev_templates)))
        train_dev_common = train_templates.intersection(dev_templates)
        dev_extra_templates = dev_templates.difference(train_templates)
        print("Train / Dev common program templates (this disregards arguments): {}".format(len(train_dev_common)))
        print("Dev extra abstract program templates: {}".format(len(dev_extra_templates)))
    print()

    template2count_sorted = sorted(train_optemplate2count.items(), key=lambda x: x[1], reverse=True)
    for i in range(0, 10):
        template, count = template2count_sorted[i]
        print("{} {}".format(template, count))
        for j in range(0, 5):
            qid = train_optemplate2qids[template][j]
            ques = train_qid2ques[qid]
            print(ques)


def write_example_programs_tsv(output_tsv_path, qid2ques, qid2nestedexp, func2qids):
    """Write example (question, program) in TSV for Google Sheets.

    To ensure diversity, we first sample 10 questions for each function type.
    """
    print("Writing examples to TSV: {}".format(output_tsv_path))
    qid_ques_programs = []
    for func, qids in func2qids.items():
        print(func)
        random.shuffle(qids)
        for i in range(0, min(20, len(qids))):
            qid = qids[i]
            qid_ques_programs.append((func, qid, qid2ques[qid], qid2nestedexp[qid]))

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



    train_qid2ques, train_optemplate2count, train_optemplate2qids, train_qid2nestedexp = read_qdmr(
        qdmr_examples=train_qdmr_examples)
    dev_qid2ques, dev_optemplate2count, dev_optemplate2qids, dev_qid2nestedexp = None, None, None, None
    if dev_qdmr_json is not None:
        dev_qid2ques, dev_optemplate2count, dev_optemplate2qids, dev_qid2nestedexp = read_qdmr(
            qdmr_examples=dev_qdmr_examples)

    train_dev_stats(train_qid2ques, train_optemplate2count, train_optemplate2qids,
                    dev_qid2ques, dev_optemplate2count, dev_optemplate2qids)

    # if output_tsv_path:
    #     write_example_programs_tsv(output_tsv_path, train_qid2ques, train_qid2nestedexp, train_func2qids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_qdmr_json")
    parser.add_argument("--dev_qdmr_json")
    parser.add_argument("--output_tsv_path")
    args = parser.parse_args()

    main(args)
