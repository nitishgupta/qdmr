# QDMR processing

## Question-decomposition to program -- `parse_dataset/parse_qdmr.py`
The script `parse_dataset/parse_qdmr.py` coverts break-csv into a json file with question-decomp
along with program (as parsed by Tomer's code) and it's nested expression.

```
# DROP
python -m parse_dataset.parse_qdmr --qdmr_csv /shared/nitishg/data/Break-dataset/QDMR-high-level/train.csv --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/train.json --dataset DROP
python -m parse_dataset.parse_qdmr --qdmr_csv /shared/nitishg/data/Break-dataset/QDMR-high-level/dev.csv --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/dev.json --dataset DROP

# HOTPOTQA
python -m parse_dataset.parse_qdmr --qdmr_csv /shared/nitishg/data/Break-dataset/QDMR-high-level/train.csv --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/HOTPOT/train.json --dataset HOTPOT
python -m parse_dataset.parse_qdmr --qdmr_csv /shared/nitishg/data/Break-dataset/QDMR-high-level/dev.csv --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/HOTPOT/dev.json --dataset HOTPOT
```

The output json file contains a list of instances as dict where each contains the following keys:
```
question_id: string
question_text: string
split: string
decomposition: string
program: List[string]  --  As parsed by Tomer's code. Linearized and has statements like "AGGREGATE['count', '#1']" 
nested_expression: List  --  nested expression for the program with resolved references and math-functions. e.g. above becomes "AGGREGATE_count" 
operators: List[string]
```

Note: If a decomposition cannot be parsed into a program, that program would contain `"None"` steps.
The equivalent `"nested_expression"` would be an empty-list.

## Utilities -- `qdmr/utils.py`
Contains tons of helper functions and classes to parse qdmr json and programs.

`QDMRExample` is a class that can store the json instance in an easy accessible manner. 

Function `read_qdmr_json_to_examples` can be used to read a json file into a `List[QDMRExample]`

Function `nested_expression_to_tree` converts a program's nested expression into a Tree representation where each node
is represented by the `Node` class. 

Function `string_arg_to_quesspan_pred` can convert the `string` arguments to the `GET_QUESTION_SPAN` predicate,
and store the underlying string argument in `Node` class' `string_arg` member. 


## DROP grammar-constrained programs -- `parse_dataset/qdmr_grammar_program.py`
By analyzing DROP-programs we come up with a language to parse DROP programs into 
(`qdmr/domain_languages/qdmr_language.py`)

The script `parse_dataset/qdmr_grammar_program.py` takes as input from the `parse_qdmr.py` script
and adds the field `typed_nested_expression` to the same json file.

## Template-based Data Split -- `parse_dataset/template_split.py`
The script `parse_dataset/template_split.py` re-splits the train/dev DROP data based on program-templates.
The split ensures that the abstract-program-templates in train and test are disjoint and makes prefers a split where
1. Num of train questions is within a tolerance (200) of the target (80% of total)
2. All predicates/functions that are in the dataset appear in train templates
3. The test set contains the maximum number of functions/predicates

```
python -m parse_dataset.template_split \
    --train_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/train.json \
    --dev_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/dev.json \
    --tmp_split_train_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/template_based_split/DROP/train.json \
    --tmp_split_test_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/template_based_split/DROP/test.json
```
