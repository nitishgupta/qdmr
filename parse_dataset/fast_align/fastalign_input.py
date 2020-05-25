from typing import List, Dict, Set, Tuple

import os
import random
import argparse
from enum import Enum

from allennlp.data.tokenizers import SpacyTokenizer
from qdmr.domain_languages.drop_language import DROPLanguage

from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, nested_expression_to_lisp, \
    get_inorder_function_list, Node, linearize_nested_expression, nested_expression_to_tree

from qdmr.data import utils
random.seed(1)

spacy_tokenizer = SpacyTokenizer()


droplanguage = DROPLanguage()

""" Convert QDMR data into fast-align input format """


class ProgramType(Enum):
    Seq2Seq = 1
    Grammar = 2


def fast_align_format(qdmr_examples: List[QDMRExample], program_type: ProgramType, inorder: bool):
    """ Make fast-align formatted data for QDMR examples.

    Parameters:
    -----------
    program_type: `ProgramType`
         Seq2Seq or Grammar based program
     inorder: `bool`
        In Seq2Seq, True: use inorder function linearization; False: bracketed linearization
    """

    question_program_tokens: List[Tuple[List[str], List[str]]] = []
    question_ids: List[str] = []

    print("Num input QDMR examples: {}".format(len(qdmr_examples)))

    for qdmr_example in qdmr_examples:
        drop_nested_expression = qdmr_example.drop_nested_expression
        if not drop_nested_expression:
            continue
        program_tree: Node = nested_expression_to_tree(drop_nested_expression, predicates_with_strings=True)
        program_lisp: str = nested_expression_to_lisp(program_tree.get_nested_expression())
        try:
            droplanguage.logical_form_to_action_sequence(program_lisp)
        except:
            continue

        question = qdmr_example.question
        query_id = qdmr_example.query_id
        question_tokens: List[str] = [t.text for t in spacy_tokenizer.tokenize(question)]

        if program_type is ProgramType.Seq2Seq:
            if inorder:
                linearized_program: List[str] = get_inorder_function_list(node=program_tree)
            else:
                linearized_program: List[str] = linearize_nested_expression(program_tree.get_nested_expression())

        elif program_type is ProgramType.Grammar:
            linearized_program: List[str] = droplanguage.logical_form_to_action_sequence(program_lisp)
            linearized_program = [action.replace(" ", "###") for action in linearized_program]

        else:
            raise NotImplementedError


        question_program_tokens.append((question_tokens, linearized_program))
        question_ids.append(query_id)

    print("Total number of programs: {}".format(len(question_ids)))

    return question_program_tokens, question_ids


def main(args):

    qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(args.qdmr_json)

    output_dir, output_filename = os.path.split(args.qdmr_json)

    for progtype in ["seq2seq", "grammar"]:
        for inorder in [True, False]:
            if progtype == "grammar" and inorder:
                continue

            if progtype == "seq2seq":
                program_type = ProgramType.Seq2Seq
            elif progtype == "grammar":
                program_type = ProgramType.Grammar
            else:
                raise NotImplementedError

            question_program_tokens, question_ids = fast_align_format(qdmr_examples=qdmr_examples,
                                                                      program_type=program_type,
                                                                      inorder=inorder)

            # Filename
            inorderstr = ""
            if progtype == "seq2seq" and inorder:
                inorderstr = ".inorder"
            # Example -- train.fast_align.seq2seq.inorder
            fastalign_outfile = output_filename[:-5] + ".fast_align" + ".{}".format(progtype) + inorderstr
            qids_outfile = fastalign_outfile + ".qids"

            fast_align_output_path = os.path.join(output_dir, fastalign_outfile)
            with open(fast_align_output_path, 'w') as outf:
                for (ques_tokens, program_tokens) in question_program_tokens:
                    out_str = " ".join(ques_tokens)
                    out_str += " ||| "
                    out_str += " ".join(program_tokens)
                    out_str += "\n"
                    outf.write(out_str)

            qids_output_path = os.path.join(output_dir, qids_outfile)
            with open(qids_output_path, 'w') as outf:
                for qid in question_ids:
                    out_str = f"{qid}\n"
                    outf.write(out_str)

            print("Outputs written to: {}".format(fast_align_output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdmr_json", required=True)

    args = parser.parse_args()

    main(args)
