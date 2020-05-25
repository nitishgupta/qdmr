#!/usr/bin/env

CONFIGFILE=training_config/qdmr_seq2seq_scratch.jsonnet
INCLUDE_PACKAGE=qdmr


DATASET_ROOT=/shared/nitishg/data/qdmr-processed/QDMR-high-level
DATASET_NAME=DROP
SPLIT_TYPE=drop-template-ds     # drop-standard or drop-template

TRAINFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/train.json
DEVFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/test.json

export EMB_DIM=100

export EPOCHS=60
export BATCH_SIZE=64
export SEED=1

export TRAIN_FILE=${TRAINFILE}
export DEV_FILE=${DEVFILE}


####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=/shared/nitishg/qdmr/semparse-gen/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}
MODEL_DIR=Seq2Seq-scratch
PD=BS_${BATCH_SIZE}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD}/S_${SEED}

#######################################################################################################################

bash scripts/allennlp/train.sh ${CONFIGFILE} \
                               ${INCLUDE_PACKAGE} \
                               ${SERIALIZATION_DIR}

