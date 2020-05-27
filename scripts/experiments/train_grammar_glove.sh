#!/usr/bin/env

CONFIGFILE=training_config/qdmr_grammar_glove.jsonnet
INCLUDE_PACKAGE=qdmr


DATASET_ROOT=/shared/nitishg/data/qdmr-processed/QDMR-high-level
DATASET_NAME=DROP/resplits
SPLIT_TYPE=full-20

TRAINFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/train.json
DEVFILE=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/dev.json

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
export SEED=1

for seed in 1 2 3 4 5
do
  for attnloss in false # true
  do
    export ATTNLOSS=${attnloss}
    export SEED=${seed}

    PD=BS_${BATCH_SIZE}/ATTNLOSS_${ATTNLOSS}
    SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD}/S_${SEED}

    allennlp train ${CONFIGFILE} --include-package ${INCLUDE_PACKAGE} -s ${SERIALIZATION_DIR} &
  done
done
