#!/usr/bin/env

CONFIGFILE=training_config/qdmr_grammar_bert.jsonnet
INCLUDE_PACKAGE=qdmr

DATASET_ROOT=/shared/nitishg/data/qdmr-processed/QDMR-high-level
DATASET_NAME=DROP/resplits
SPLIT_TYPE=full-20

TRAINFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/train.json
DEVFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/dev.json

export ATTNLOSS=true

export EPOCHS=100
export BATCH_SIZE=16
export SEED=3

export TRAIN_FILE=${TRAINFILE}
export DEV_FILE=${DEVFILE}


####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=/shared/nitishg/qdmr/semparse-gen/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}
MODEL_DIR=Grammar-bert
PD=BS_${BATCH_SIZE}/ATTNLOSS_${ATTNLOSS}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD}/S_${SEED}

# SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/test

#######################################################################################################################

bash scripts/allennlp/train.sh ${CONFIGFILE} \
                               ${INCLUDE_PACKAGE} \
                               ${SERIALIZATION_DIR}