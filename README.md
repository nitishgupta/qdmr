# QDMR processing

## Question-decomposition to program
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

