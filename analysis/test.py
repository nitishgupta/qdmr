import re

from qdmr.domain_languages.drop_language import DROPLanguage
from qdmr.data.utils import read_qdmr_json_to_examples, nested_expression_to_lisp, nested_expression_to_tree, QDMRExample

language = DROPLanguage()

train_qdmr = "/shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/drop-programs/train.json"
dev_qdmr = "/shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/drop-programs/dev.json"

train = read_qdmr_json_to_examples(train_qdmr)
dev = read_qdmr_json_to_examples(dev_qdmr)

all_examples = train + dev
print("Total: {}".format(len(all_examples)))

parsable = 0
unparsable = 0
for example in all_examples:
    example: QDMRExample = example
    nested_expr = example.drop_nested_expression
    if not nested_expr:
        continue

    program_tree = nested_expression_to_tree(nested_expr, predicates_with_strings=True)
    lisp = nested_expression_to_lisp(program_tree.get_nested_expression())
    try:
        language.logical_form_to_action_sequence(lisp)
        parsable += 1
    except:
        unparsable += 1

print("Parsable:{}".format(parsable))
print("Grammar unparsable:{}".format(unparsable))


