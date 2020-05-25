import os
import json
from qdmr.data.utils import read_qdmr_json_to_examples, nested_expression_to_tree


from typing import List, Dict
from collections import defaultdict
import random
import argparse

from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, Node

from qdmr.data import utils
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.tokenizers.token import Token

random.seed(28)
spacy_tokenizer = SpacyTokenizer()

"""
This script writes examples of programs for each predicate in the dataset to a tsv file.
The output TSV can be used to create a google-sheet for analysis. 
"""


def tokenize(text: str) -> List[str]:
    tokens: List[Token] = spacy_tokenizer.tokenize(text)
    return [t.text for t in tokens]


def get_question_args(node: Node):
    string_args: List[str] = []
    for child in node.children:
        string_args.extend(get_question_args(child))

    if node.predicate == "GET_QUESTION_SPAN":
        string_args.append(node.string_arg)

    return string_args


def arg_in_question_perc(question_tokens, arg_tokens):
    ques_tokens = set(question_tokens)
    arg_tokens = set(arg_tokens)
    if "REF" in arg_tokens:
        arg_tokens.remove("REF")
    if "#" in arg_tokens:
        arg_tokens.remove("#")
    if not arg_tokens:
        return 0.0
    return float(len(arg_tokens.intersection(ques_tokens)))/len(arg_tokens)


def analyze_question_string_arg(qdmr_examples: List[QDMRExample]):
    """Statisitcs for match between question-string-arg and question itself."""

    total_overlap = 0.0
    total_args = 0
    total_ques = 0

    for qdmr in qdmr_examples:
        question = qdmr.question
        program_tree: Node = qdmr.program_tree
        if program_tree is None:
            continue

        total_ques += 1
        question_tokens = tokenize(question)
        string_args = get_question_args(program_tree)
        all_arg_tokens = [tokenize(a) for a in string_args]

        q_overlap_ratio = sum([arg_in_question_perc(question_tokens, arg_tokens) for arg_tokens in all_arg_tokens])
        overlap_ratio = q_overlap_ratio/len(all_arg_tokens)

        total_overlap += q_overlap_ratio
        total_args += len(string_args)

        print(question)
        print(program_tree.get_nested_expression_with_strings())
        print(string_args)
        print(f"average-overlap: {overlap_ratio}")
        print()

    average_overlap = float(total_overlap)/total_args
    print(f"Total ques: {total_ques}  Total string-args: {total_args}")
    print(f"Average overlap: {average_overlap}")




def main(args):
    qdmr_json = args.qdmr_json

    qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(qdmr_json)

    analyze_question_string_arg(qdmr_examples)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdmr_json")
    args = parser.parse_args()

    main(args)
