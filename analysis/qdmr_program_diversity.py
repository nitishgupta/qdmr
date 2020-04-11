from typing import List, Dict
from collections import defaultdict
import random
import argparse

from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample

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
    qdmr_json = args.qdmr_json

    output_tsv_path = args.output_tsv_path

    qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(qdmr_json)

    qid2qdmrexample, pred2qids = get_maps(qdmr_examples)

    write_example_programs_tsv(output_tsv_path, qid2qdmrexample, pred2qids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdmr_json")
    parser.add_argument("--output_tsv_path")
    args = parser.parse_args()

    main(args)
