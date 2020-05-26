from typing import List
import os
import argparse

from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, write_qdmr_examples_to_json, Node

def is_select_question(root: Node):
    return root.predicate == "SELECT" and len(root.children) == 1 and root.children[0].predicate == "GET_QUESTION_SPAN"


def remove_select_questions(qdmr_examples: List[QDMRExample]) -> List[QDMRExample]:
    filtered_examples = []
    for example in qdmr_examples:
        if example.program_tree and example.drop_nested_expression:
            if not is_select_question(example.program_tree):
                filtered_examples.append(example)
    return filtered_examples


def main(args):

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for filename in "train.json", "dev.json", "test.json":
        print("\nprocessing: {}".format(filename))
        qdmr_file = os.path.join(input_dir, filename)
        if not os.path.exists(qdmr_file):
            continue

        qdmr_examples = read_qdmr_json_to_examples(qdmr_file)
        print("Original examples: {}".format(len(qdmr_examples)))

        filtered_qdmr_examples = remove_select_questions(qdmr_examples)
        print("after filtering: {}".format(len(filtered_qdmr_examples)))

        write_qdmr_examples_to_json(filtered_qdmr_examples, os.path.join(output_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    main(args)
