from typing import List, Dict, Set, Tuple

import argparse
import os
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.predictors import Predictor
from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, write_qdmr_examples_to_json
import allennlp_models.syntax.constituency_parser

print("Loading constituency parser ...")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
                                cuda_device=0)

from qdmr.predictors.consituency_parser import ConstituencyParserPredictorWhiteSpace
predictor_ours = ConstituencyParserPredictorWhiteSpace(predictor._model, predictor._dataset_reader)
spacy_tokenizer = SpacyTokenizer()
print("Loaded!")


def get_parse_spans(node, spans):
    if "root" in node:
        spans = get_parse_spans(node["root"], spans)
    else:
        nodelabel = node["nodeType"]
        spans.append((node["start"], node["end"], node["word"], nodelabel))
        if 'children' in node:
            for child in node['children']:
                spans = get_parse_spans(child, spans)
    return spans


def get_all_spans(text: str):
    output: Dict = predictor_ours.predict(sentence=text)
    tree = output['hierplane_tree']
    spans = get_parse_spans(tree, [])
    tokens = output['tokens']
    return spans, tokens


def add_spans_to_qdmrs(qdmr_examples: List[QDMRExample]):
    """ Add to Question Tokens to QDMRExample. """
    for qdmr_example in qdmr_examples:
        question = qdmr_example.question
        if "question_tokens" not in qdmr_example.extras:
            question_tokens: List[str] = [t.text for t in spacy_tokenizer.tokenize(question)]
            qdmr_example.extras["question_tokens"] = question_tokens
        question_tokens = qdmr_example.extras["question_tokens"]

        whitespace_tokenized_question: str = " ".join(question_tokens)
        spans, tokens = get_all_spans(whitespace_tokenized_question)

        assert question_tokens == tokens, "Tokenization mismatch"

        unique_spans = set()
        for span in spans:
            start, end = span[0], span[1]
            if start == 0 and end == len(question_tokens) - 1:
                continue
            unique_spans.add((start, end))
        spans = sorted(list(unique_spans), key=lambda x: x[0])
        qdmr_example.extras["question_spans"] = spans

    return qdmr_examples


def read_list_string_file(filepath: str):
    with open(filepath) as f:
        lines = f.readlines()
    return lines


def main(args):
    # get_all_spans("How many more points did the Pittsburgh Steelers score than the Washington Redskins in 2008?")

    input_dir = args.input_dir

    print("Reading data from : {}".format(input_dir))
    train_qdmrs: List[QDMRExample] = read_qdmr_json_to_examples(os.path.join(input_dir, "train.json"))
    dev_qdmrs: List[QDMRExample] = read_qdmr_json_to_examples(os.path.join(input_dir, "dev.json"))

    print("Adding constituent spans ...")
    updated_train_qdmrs = add_spans_to_qdmrs(train_qdmrs)
    updated_dev_qdmrs = add_spans_to_qdmrs(dev_qdmrs)

    print("Writing data to : {}".format(input_dir))
    write_qdmr_examples_to_json(updated_train_qdmrs, os.path.join(input_dir, "train.json"))
    write_qdmr_examples_to_json(updated_dev_qdmrs, os.path.join(input_dir, "dev.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)

    args = parser.parse_args()

    main(args)
