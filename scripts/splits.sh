#!/usr/bin/env

ROOT_DIR=/shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP

STD=new-splits/drop-standard
STD_DS=new-splits/drop-standard-ds

TMP=new-splits/drop-template-manual
TMP_DS=new-splits/drop-template-manual-ds

mkdir -p ${ROOT_DIR}/${STD}
python -m parse_dataset.split_qdmr_data \
  --train_qdmr_json ${ROOT_DIR}/drop-programs/train.json \
  --dev_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/drop-programs/dev.json \
  --out_dir ${ROOT_DIR}/${STD} \
  --split std > ${ROOT_DIR}/${STD}/stats.txt


mkdir -p ${ROOT_DIR}/${STD_DS}
python -m parse_dataset.split_qdmr_data \
  --train_qdmr_json ${ROOT_DIR}/drop-programs/train.json \
  --dev_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/drop-programs/dev.json \
  --out_dir ${ROOT_DIR}/${STD_DS} \
  --split std \
  --downsample > ${ROOT_DIR}/${STD_DS}/stats.txt

mkdir -p ${ROOT_DIR}/${TMP}
python -m parse_dataset.split_qdmr_data \
  --train_qdmr_json ${ROOT_DIR}/drop-programs/train.json \
  --dev_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/drop-programs/dev.json \
  --out_dir ${ROOT_DIR}/${TMP} \
  --split tmp \
  --manual_test > ${ROOT_DIR}/${TMP}/stats.txt

mkdir -p ${ROOT_DIR}/${TMP_DS}
python -m parse_dataset.split_qdmr_data \
  --train_qdmr_json ${ROOT_DIR}/drop-programs/train.json \
  --dev_qdmr_json /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/drop-programs/dev.json \
  --out_dir ${ROOT_DIR}/${TMP_DS} \
  --split tmp \
  --manual_test \
  --downsample > ${ROOT_DIR}/${TMP_DS}/stats.txt

