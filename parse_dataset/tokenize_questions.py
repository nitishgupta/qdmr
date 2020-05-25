from typing import List, Dict, Set, Tuple

import os
import argparse

from allennlp.data.tokenizers import SpacyTokenizer
from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, write_qdmr_examples_to_json

spacy_tokenizer = SpacyTokenizer()


def add_tokenized_questions(qdmr_examples: List[QDMRExample]):
    """ Add to Question Tokens to QDMRExample. """

    for qdmr_example in qdmr_examples:
        question = qdmr_example.question
        question_tokens: List[str] = [t.text for t in spacy_tokenizer.tokenize(question)]
        qdmr_example.extras["question_tokens"] = question_tokens

    return qdmr_examples


def read_list_string_file(filepath: str):
    with open(filepath) as f:
        lines = f.readlines()
    return lines


def main(args):

    for filename in ["train", "dev", "test"]:
        input_file = os.path.join(args.input_dir, filename + ".json")
        qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(input_file)

        updated_qdmr_examples = add_tokenized_questions(qdmr_examples=qdmr_examples)

        print("Written outputs back to : {}".format(input_file))
        write_qdmr_examples_to_json(updated_qdmr_examples, input_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)

    args = parser.parse_args()

    main(args)
