#!/usr/bin/env

drop_raw_dir=/shared/nitishg/data/drop/raw
break_data_dir=/shared/nitishg/data/break-dataset
drop_qdmr_dir=/shared/nitishg/data/drop/iclr21/qdmr-processed

mkdir -p ${drop_qdmr_dir}
mkdir -p ${drop_qdmr_dir}/original-data
mkdir -p ${drop_qdmr_dir}/drop-programs


# Parse QDMR DROP data from CSV into JSON
python -m parse_dataset.parse_qdmr \
    --qdmr_csv ${break_data_dir}/QDMR-high-level/train.csv \
    --dataset DROP \
    --qdmr_json ${drop_qdmr_dir}/original-data/train.json

python -m parse_dataset.parse_qdmr \
    --qdmr_csv ${break_data_dir}/QDMR-high-level/dev.csv \
    --dataset DROP \
    --qdmr_json ${drop_qdmr_dir}/original-data/dev.json


# DROP grammar-contrained programs
python -m parse_dataset.qdmr_grammar_program \
    --qdmr_json ${drop_qdmr_dir}/original-data/train.json

python -m parse_dataset.qdmr_grammar_program \
    --qdmr_json ${drop_qdmr_dir}/original-data/dev.json


# DROP manual program-transformations for executable programs
python -m parse_dataset.drop_grammar_program \
    --qdmr_json ${drop_qdmr_dir}/original-data/train.json \
    --drop_json ${drop_raw_dir}/drop_dataset_train.json \
    --qdmr_split train \
    --output_json ${drop_qdmr_dir}/drop-programs/train.json

python -m parse_dataset.drop_grammar_program \
    --qdmr_json ${drop_qdmr_dir}/original-data/dev.json \
    --drop_json ${drop_raw_dir}/drop_dataset_dev.json \
    --qdmr_split dev \
    --output_json ${drop_qdmr_dir}/drop-programs/dev.json

