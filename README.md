# QDMR processing

## Question-decomposition to program -- `parse_dataset/parse_qdmr.py`
The script `parse_dataset/parse_qdmr.py` coverts break-csv into a json file with question-decomp
along with program (as parsed by Tomer's code) and it's nested expression.

```
## DROP ##
python -m parse_dataset.parse_qdmr \
    --qdmr_csv /shared/nitishg/data/Break-dataset/QDMR-high-level/train.csv \
    --dataset DROP \
    --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/original-data/train.json
 
python -m parse_dataset.parse_qdmr \
    --qdmr_csv /shared/nitishg/data/Break-dataset/QDMR-high-level/dev.csv \
    --dataset DROP \
    --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/original-data/dev.json 

## HOTPOTQA ##
python -m parse_dataset.parse_qdmr \
    --qdmr_csv /shared/nitishg/data/Break-dataset/QDMR-high-level/train.csv \
    --dataset HOTPOT \
    --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/HOTPOT/original-data/train.json 

python -m parse_dataset.parse_qdmr \
    --qdmr_csv /shared/nitishg/data/Break-dataset/QDMR-high-level/dev.csv \
    --dataset HOTPOT \
    --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/HOTPOT/original-data/dev.json 
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
and adds the following field to the same json file: ```typed_nested_expression``` 

```
python -m parse_dataset.qdmr_grammar_program \
    --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/original-data/train.json

python -m parse_dataset.qdmr_grammar_program \
    --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/original-data/dev.json
```

## Splits for evaluating Semantic Parser Generalization

### Standard train-dev-test split -- `parse_dataset/standard_split.py`
Splits the original-data's train split into a 90/10 train/dev split. 
Uses the original-data's dev split as the new test split.

```
python -m parse_dataset.standard_split \
    --train_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/original-data/train.json \
    --dev_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/original-data/dev.json \
    --std_split_dir  /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/standard-split
``` 


### Template-based Data Split -- `parse_dataset/template_split.py`
The script `parse_dataset/template_split.py` re-splits the train/dev DROP data based on program-templates.
The split ensures that the abstract-program-templates in train and test are disjoint and makes prefers a split where
1. Num of train questions is within a tolerance (200) of the target (80% of total)
2. All predicates/functions that are in the dataset appear in train templates
3. The test set contains the maximum number of functions/predicates

This script makes 4 splits, `train.json`, `dev.json`, `dev-out.json` and `test.json` -- where 
`dev.json` is a in-domain dev set (10% of train) and 
`dev-out.json` is a dev set from the test templates (20% of test)

```
python -m parse_dataset.template_split \
    --train_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP//original-data/train.json \
    --dev_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/original-data/dev.json \
    --tmp_split_dir /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/template-split
```

## TSV Examples for Grammar Programs -- `analysis/qdmr_program_diversity.py`
Write examples of programs for each predicate in the dataset to a TSV file.
```
python -m analysis.qdmr_program_diversity \
  --qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/train.json \
  --output_tsv_path /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/examples.tsv
```
