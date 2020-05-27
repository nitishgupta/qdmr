#!/usr/bin/env

ROOT_DIR=/shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP

SPLITS_DIR=resplits-bkp

#mkdir -p ${ROOT_DIR}/${SPLITS_DIR}/full
#python -m parse_dataset.split_qdmr_data \
#  --drop_dir ${ROOT_DIR}/drop-programs \
#  --output_dir ${ROOT_DIR}/${SPLITS_DIR}/full > ${ROOT_DIR}/${SPLITS_DIR}/full/stats.txt


#for DS_LIMIT in 50 40 30 20 10
#do
#  mkdir -p ${ROOT_DIR}/${SPLITS_DIR}/full-${DS_LIMIT}
#  python -m parse_dataset.split_qdmr_data \
#    --drop_dir ${ROOT_DIR}/drop-programs \
#    --output_dir ${ROOT_DIR}/${SPLITS_DIR}/full-${DS_LIMIT} \
#    --downsample \
#    --ds_template_limit ${DS_LIMIT} > ${ROOT_DIR}/${SPLITS_DIR}/full-${DS_LIMIT}/stats.txt
#done


for DS_LIMIT in 50 40 30 20 10
do
  mkdir -p ${ROOT_DIR}/${SPLITS_DIR}/full-${DS_LIMIT}
  python -m parse_dataset.update_extras \
    --parent_dir /shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/drop-programs \
    --output_dir ${ROOT_DIR}/${SPLITS_DIR}/full-${DS_LIMIT}
done