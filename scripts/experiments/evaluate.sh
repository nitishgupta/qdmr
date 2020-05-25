#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=qdmr
export GPU=0
export BEAMSIZE=1

# SAVED MODEL

DATASET_ROOT=/shared/nitishg/data/qdmr-processed/QDMR-high-level
DATASET_NAME=DROP/splits

for attn in true false
do
  for SPLIT_TYPE in drop-template-manual drop-template-manual-ds drop-standard drop-standard-ds
  do
    # MODEL_DIR=/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/splits/${SPLIT_TYPE}/Seq2Seq-glove/BS_4/INORDER_true/ATTNLOSS_${attn}/S_1337
    MODEL_DIR=/shared/nitishg/qdmr/semparse-gen/checkpoints/DROP/splits/${SPLIT_TYPE}/Grammar-glove/BS_4/ATTNLOSS_${attn}/S_1337
    MODEL_TAR=${MODEL_DIR}/model.tar.gz
    PREDICTION_DIR=${MODEL_DIR}/predictions
    mkdir ${PREDICTION_DIR}

    DEVFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/dev.json
    METRICS_FILE=${PREDICTION_DIR}/dev_metrics.json
    allennlp evaluate --output-file ${METRICS_FILE} \
                      --cuda-device ${GPU} \
                      --include-package ${INCLUDE_PACKAGE} \
                      ${MODEL_TAR} ${DEVFILE} &

    TESTFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/test.json
    METRICS_FILE=${PREDICTION_DIR}/test_metrics.json
    allennlp evaluate --output-file ${METRICS_FILE} \
                      --cuda-device ${GPU} \
                      --include-package ${INCLUDE_PACKAGE} \
                      ${MODEL_TAR} ${TESTFILE}
  done
done
