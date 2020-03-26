import os
import json
import pandas as pd
import argparse
from parse_dataset.parse import QDMRProgramBuilder, qdmr_program_to_nested_expression


def convert_to_json(qdmr_csv: str, qdmr_json: str, dataset: str = None):
    """ Convert QDMR csv data into json format. Also parse the decomposition into a linearized program.

    Tomer gave this code in a Jupyter notebook.

    Parameters
    ---------
    qdmr_csv : ``str``
        Input csv path
    qdmr_json : ``str``
        Output json path
    dataset : ``str`` optional (default = None)
        If supplied, question_id will be filtered based on the presence of this token. Used to only parse a particular
        dataset within QDMR. e.g. "DROP"
    """

    df = pd.read_csv(qdmr_csv)
    decompositions = df['decomposition']

    total_questions = 0
    ques_with_programs = 0
    num_nested_exprs = 0

    json_objects = []

    for i in range(len(decompositions)):
        question_id = df.loc[i, 'question_id']
        question_text = df.loc[i, 'question_text']
        decomposition = df.loc[i, 'decomposition']
        split = df.loc[i, 'split']

        if dataset is not None and dataset not in question_id:
            continue

        builder = QDMRProgramBuilder(decomposition)

        # Copied this try-block from Tomer's code but it is broken. program and operators are always populated. Usually
        #  one or more of the tokens in program is "None" when decomposition parsing fails. This will need to be taken
        #  care of post-hoc.
        try:
            builder.build()
            program = [str(step) for step in builder.steps]
            operators = [str(op) for op in builder.operators]
        except:
            program = ['ERROR']
            operators = ['ERROR']

        if "None" not in program:
            ques_with_programs += 1
            try:
                # TODO(nitish): This fails for argument "#1 that were", etc. but only found 2-5 cases in DROP and HOTPOT
                nested_expression = qdmr_program_to_nested_expression(qdmr_program=program)
                num_nested_exprs += 1
            except:
                nested_expression = []
        else:
            nested_expression = []

        total_questions += 1

        json_object = {
            "question_id": question_id,
            "question_text": question_text,
            "split": split,
            "decomposition": decomposition,
            "program": program,
            "nested_expression": nested_expression,
            "operators": operators
        }
        json_objects.append(json_object)

    with open(qdmr_json, 'w') as outf:
        json.dump(json_objects, outf, indent=4)

    print("Total questions: {}  w/ programs: {}".format(total_questions, ques_with_programs))
    print("Total nested expressions: {}".format(num_nested_exprs))


def main(args):
    qdmr_csv = args.qdmr_csv
    qdmr_json = args.qdmr_json
    dataset = args.dataset

    output_dir = os.path.split(qdmr_json)[0]
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    convert_to_json(qdmr_csv, qdmr_json, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdmr_csv")
    parser.add_argument("--qdmr_json")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    main(args)
