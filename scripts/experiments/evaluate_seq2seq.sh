#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=qdmr
export GPU=0
export BEAMSIZE=1

# SAVED MODEL
MODEL_DIR=/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/template-split/Seq2Seq/BS_64/S_1
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

DATASET_ROOT=/shared/nitishg/data/qdmr-processed/QDMR-high-level
DATASET_NAME=DROP
SPLIT_TYPE=template-split
DEVFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/dev-out.json

METRICS_FILE=${PREDICTION_DIR}/dev-out_metrics.json

allennlp evaluate --output-file ${METRICS_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${DEVFILE}