from typing import List, Dict, Tuple, Set
from collections import defaultdict
import random
import argparse
import os

from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, get_inorder_function_list_from_template

from qdmr.data import utils

random.seed(28)


"""
This script writes examples of programs for each predicate in the dataset to a tsv file.
The output TSV can be used to create a google-sheet for analysis. 
"""


def get_maps(qdmr_examples: List[QDMRExample]):
    """ This data is processed using parse_dataset.parse_qdmr and keys in this json can be glanced at from there.

    The nested_expression in the data makes life easier since functions are already normalized to their min/max, add/sub
    identifier.

    This function mainly reads relevant
    """

    qid2qdmrexample = {}
    for qdmr_example in qdmr_examples:
        qid2qdmrexample[qdmr_example.query_id] = qdmr_example
    qid2template = {}
    pred2qids = defaultdict(list)
    template2qids = defaultdict(list)
    complexity2templates = defaultdict(list)
    template2count = defaultdict(int)
    template2funcs = defaultdict(set)
    for qdmr_example in qdmr_examples:
        qid = qdmr_example.query_id
        if not qdmr_example.program_tree:
            continue
        nested_expr = qdmr_example.program_tree.get_nested_expression()
        func_set, template = utils.convert_nestedexpr_to_tuple(nested_expr)
        complexity = len(get_inorder_function_list_from_template(template))
        if template not in complexity2templates[complexity]:
            complexity2templates[complexity].append(template)
        for pred in func_set:
            pred2qids[pred].append(qid)
        qid2template[qid] = template
        template2qids[template].append(qid)
        template2count[template] += 1
        template2funcs[template].update(func_set)

    return qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs


def write_example_predicate2questions_tsv(predicate_tsv: str,
                                          qid2qdmrexample: Dict[str, QDMRExample],
                                          pred2qids: Dict[str, List[str]]):
    """Write example (question, program) in TSV for Google Sheets.

    To ensure diversity, we first sample 10 questions for each function type.
    """
    print("Writing examples to predicate-TSV: {}".format(predicate_tsv))
    preds_list = list(pred2qids.keys())
    preds_list = sorted(preds_list)

    qid_ques_programs = []
    for func in preds_list:
        qids = pred2qids[func]
        print(func)
        random.shuffle(qids)
        for i in range(0, min(15, len(qids))):
            qid = qids[i]
            qdmr_example = qid2qdmrexample[qid]
            nested_expr = qdmr_example.program_tree.get_nested_expression_with_strings()
            qid_ques_programs.append((func,
                                      qid,
                                      qdmr_example.question,
                                      nested_expr))

    with open(predicate_tsv, 'w') as outf:
        outf.write(f"Function\tQueryID\tQuestion\tProgram\n")
        # random.shuffle(qid_programs)
        for (func, qid, ques, program) in qid_ques_programs:
            out_str = f"{func}\t{qid}\t{ques}\t{program}\n"
            outf.write(out_str)

    print("Total examples written: {}".format(len(qid_ques_programs)))
    print(predicate_tsv)


def write_example_template2questions_tsv(template_tsv: str,
                                         qid2qdmrexample: Dict[str, QDMRExample],
                                         template2qids: Dict[Tuple, List[str]],
                                         complexity2templates: Dict[int, Set[Tuple]]):
    """Write example (question, program) in TSV for Google Sheets.

    To ensure diversity, we first sample 10 questions for each function type.
    """
    print("Writing examples to predicate-TSV: {}".format(template_tsv))
    sorted_complexity = sorted(complexity2templates.items(), key=lambda x: x[0])
    # Arranging templates in increasing order of complexity
    templates_list = []
    for complexity, templates in sorted_complexity:
        templates_list.extend(list(templates))

    qid_ques_programs = []
    for template in templates_list:
        qids = template2qids[template]
        random.shuffle(qids)
        for i in range(0, min(3, len(qids))):
            qid = qids[i]
            qdmr_example = qid2qdmrexample[qid]
            nested_expr = qdmr_example.program_tree.get_nested_expression_with_strings()
            qid_ques_programs.append((template,
                                      qid,
                                      qdmr_example.question,
                                      nested_expr))

    with open(template_tsv, 'w') as outf:
        outf.write(f"Template\tQueryID\tQuestion\tProgram\n")
        for (func, qid, ques, program) in qid_ques_programs:
            out_str = f"{func}\t{qid}\t{ques}\t{program}\n"
            outf.write(out_str)
    print("Total examples written: {}".format(len(qid_ques_programs)))
    print(template_tsv)


def main(args):
    qdmr_json = args.qdmr_json

    output_dir, json_filename = os.path.split(qdmr_json)
    filename_root = json_filename.split(".")[0]
    predicate_tsv = os.path.join(output_dir, filename_root + "_predicate.tsv")
    template_tsv = os.path.join(output_dir, filename_root + "_template.tsv")

    qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(qdmr_json)

    qid2qdmrexample, qid2template, pred2qids, template2qids, complexity2templates, template2count, template2funcs = \
        get_maps(qdmr_examples)

    write_example_predicate2questions_tsv(predicate_tsv, qid2qdmrexample, pred2qids)

    write_example_template2questions_tsv(template_tsv, qid2qdmrexample, template2qids, complexity2templates)

    templatecount2count = defaultdict(int)
    for _, count in template2count.items():
        templatecount2count[count] += 1

    sorted_count = sorted(templatecount2count.items(), key=lambda x: x[1], reverse=True)
    print(" ".join([str(y) + "|" + str(x) for x, y in sorted_count]))

    sorted_templatecount = sorted(template2count.items(), key=lambda x: x[1], reverse=True)
    print(" ".join([str(x) for _, x in sorted_templatecount]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdmr_json")
    args = parser.parse_args()

    main(args)
