#!/usr/bin/env

CONFIGFILE=training_config/qdmr_grammar_glove.jsonnet
INCLUDE_PACKAGE=qdmr


DATASET_ROOT=/shared/nitishg/data/qdmr-processed/QDMR-high-level
DATASET_NAME=DROP/new-splits
SPLIT_TYPE=drop-template-manual-ds

TRAINFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/train.json
DEVFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/test.json

export GLOVE=/shared/embeddings/glove/glove.6B.100d.txt
export GLOVE_EMB_DIM=100

export ATTNLOSS=false

export EPOCHS=100
export BATCH_SIZE=4
export SEED=1337

export TRAIN_FILE=${TRAINFILE}
export DEV_FILE=${DEVFILE}


####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=/shared/nitishg/qdmr/semparse-gen/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}
MODEL_DIR=Grammar-glove
PD=BS_${BATCH_SIZE}/ATTNLOSS_${ATTNLOSS}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD}/S_${SEED}

# SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/test

#######################################################################################################################

#bash scripts/allennlp/train.sh ${CONFIGFILE} \
#                               ${INCLUDE_PACKAGE} \
#                               ${SERIALIZATION_DIR}


export BATCH_SIZE=16
export SEED=1337

for attnloss in true # false
do
  export ATTNLOSS=${attnloss}

  PD=BS_${BATCH_SIZE}/ATTNLOSS_${ATTNLOSS}
  SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD}/S_${SEED}_D

  allennlp train ${CONFIGFILE} --include-package ${INCLUDE_PACKAGE} -s ${SERIALIZATION_DIR} &
done
