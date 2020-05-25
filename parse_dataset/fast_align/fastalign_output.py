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


def parse_fast_align_output(fast_align_output: str) -> List[Tuple[int, int]]: # List[List[int]]:
    """ Fast-align output string in "input_1-output_token_idx input_2-output_token_idx .... "
        We parse in `reverse` mode in which each input token is aligned to one output token.

        We would return a list of input tokens for each output token -- List[List[int]] -- where the outer list should
        have the same length as the linearized program
    """
    input2output_alignment = []
    max_output_idx = -1
    for input in fast_align_output.split(" "):
        input_token, output_token = int(input.split("-")[0]), int(input.split("-")[1])
        max_output_idx = max(output_token, max_output_idx)
        input2output_alignment.append((input_token, output_token))

    # alignments_to_input = [[] for _ in range(max_output_idx + 1)]
    # for input_token, output_token in input_output_alignment:
    #     alignments_to_input[output_token].append(input_token)

    return input2output_alignment


def add_fastalign_to_qdmr(qdmr_examples: List[QDMRExample], progtype: str, inorder: bool, qids: List[str],
                          fast_align_outputs: List[str]):

    """ Add to QDMRExample the aligment info from Fastalign


    qdmr_examples:
        Input examples
    program_type:
        Kind of program is fast alignment done for
    inorder:
        If Seq2Seq, is the program inorder-function-list (True) or bracketed-linearization (False)
    qids:
        List of qids in order for fast-align
    fast_align_output:
        List of fast-align output string in "input_1-output_token_idx input_2-output_token_idx .... "
        WE parse in `reverse` mode which has an output-token aligned for each input token
    """

    if progtype == "seq2seq":
        program_type = ProgramType.Seq2Seq
    elif progtype == "grammar":
        program_type = ProgramType.Grammar
    else:
        raise NotImplementedError

    import pdb

    qid2fastoutput = {qid.strip(): fastalign.strip() for qid, fastalign in zip(qids, fast_align_outputs)}

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

        if program_type is ProgramType.Seq2Seq:
            if inorder:
                linearized_program: List[str] = get_inorder_function_list(node=program_tree)
            else:
                linearized_program: List[str] = linearize_nested_expression(program_tree.get_nested_expression())
        elif program_type is ProgramType.Grammar:
            linearized_program: List[str] = droplanguage.logical_form_to_action_sequence(program_lisp)
            # linearized_program = [action.replace(" ", "###") for action in linearized_program]
        else:
            raise NotImplementedError

        # For every action, a list of input_tokens in gets aligned to
        program_token_2_input_alignment: List[List[int]] = [[] for _ in range(len(linearized_program))]
        fast_align_output = qid2fastoutput[qdmr_example.query_id]
        input2output_alignment: List[Tuple[int, int]] = parse_fast_align_output(fast_align_output)
        for input_idx, output_idx in input2output_alignment:
            program_token_2_input_alignment[output_idx].append(input_idx)

        assert len(program_token_2_input_alignment) == len(linearized_program)

        inorderstr = ""
        if progtype == "seq2seq" and inorder:
            inorderstr = ".inorder"

        extras_key = "fastalign" + ".{}".format(progtype) + inorderstr
        qdmr_example.extras[extras_key] = program_token_2_input_alignment

        question = qdmr_example.question
        question_tokens: List[str] = [t.text for t in spacy_tokenizer.tokenize(question)]
        qdmr_example.extras["question_tokens"] = question_tokens

    return qdmr_examples


def read_list_string_file(filepath: str):
    with open(filepath) as f:
        lines = f.readlines()
    return lines


def main(args):

    for progtype in ["seq2seq", "grammar"]:
        qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(args.qdmr_json)

        output_dir, output_filename = os.path.split(args.qdmr_json)

        for inorder in [True, False]:
            if progtype == "grammar" and inorder:
                continue

            # Fast align filename
            inorderstr = ""
            if progtype == "seq2seq" and inorder:
                inorderstr = ".inorder"

            # Example -- train.fast_align.seq2seq.inorder
            fastalign_root = output_filename[:-5] + ".fast_align" + ".{}".format(progtype) + inorderstr

            fastalign_outfile = fastalign_root + ".output"
            qids_file = fastalign_root + ".qids"

            print("Processing: {}".format(fastalign_outfile))

            fast_align_output_path = os.path.join(output_dir, fastalign_outfile)
            fast_align_outputs = read_list_string_file(fast_align_output_path)

            qids_output_path = os.path.join(output_dir, qids_file)
            qids = read_list_string_file(qids_output_path)

            updated_qdmr_examples = add_fastalign_to_qdmr(qdmr_examples=qdmr_examples, progtype=progtype,
                                                          inorder=inorder, qids=qids,
                                                          fast_align_outputs=fast_align_outputs)

            print("Written outputs back to : {}".format(args.qdmr_json))
            utils.write_qdmr_examples_to_json(updated_qdmr_examples, args.qdmr_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdmr_json", required=True)

    args = parser.parse_args()

    main(args)
