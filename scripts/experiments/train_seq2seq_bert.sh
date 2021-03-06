#!/usr/bin/env

CONFIGFILE=training_config/qdmr_seq2seq_bert.jsonnet
INCLUDE_PACKAGE=qdmr

DATASET_ROOT=/shared/nitishg/data/qdmr-processed/QDMR-high-level
DATASET_NAME=DROP/resplits
SPLIT_TYPE=full-20

TRAINFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/train.json
DEVFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/dev.json

export INORDER=true
export ATTNLOSS=false
export ATTNSPANS=false

export EPOCHS=100
export BATCH_SIZE=16
export SEED=3

export TRAIN_FILE=${TRAINFILE}
export DEV_FILE=${DEVFILE}


####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=/shared/nitishg/qdmr/semparse-gen/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}
MODEL_DIR=Seq2Seq-bert
PD=BS_${BATCH_SIZE}/INORDER_${INORDER}/ATTNLOSS_${ATTNLOSS}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD}/S_${SEED}

# SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/test
#######################################################################################################################

bash scripts/allennlp/train.sh ${CONFIGFILE} \
                               ${INCLUDE_PACKAGE} \
                               ${SERIALIZATION_DIR}

