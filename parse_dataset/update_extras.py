from typing import List, Dict, Set, Tuple

import argparse
import os
from qdmr.data.utils import read_qdmr_json_to_examples, QDMRExample, write_qdmr_examples_to_json


def main(args):
    parent_dir = args.parent_dir
    output_dir = args.output_dir

    # Parent dir should contain train.json and dev.json
    print("Reading data from : {}".format(parent_dir))
    train_qdmrs: List[QDMRExample] = read_qdmr_json_to_examples(os.path.join(parent_dir, "train.json"))
    dev_qdmrs: List[QDMRExample] = read_qdmr_json_to_examples(os.path.join(parent_dir, "dev.json"))
    parent_qdmrs = train_qdmrs + dev_qdmrs
    qid2extras = {example.query_id: example.extras for example in parent_qdmrs}
    print("Total parent qdmrs : ".format(len(parent_qdmrs)))

    print("Output dir : {}".format(output_dir))
    for filename in ["train.json", "dev.json", "heldout_test.json", "indomain_skewed_test.json",
                     "indomain_unbiased_test.json"]:
        print("Updating extras for : {}".format(filename))
        qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(os.path.join(output_dir, filename))
        for qdmr_example in qdmr_examples:
            parent_extras: Dict = qid2extras[qdmr_example.query_id]
            qdmr_example.extras.update(parent_extras)
        write_qdmr_examples_to_json(qdmr_examples, os.path.join(output_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    main(args)
