#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=qdmr
export GPU=0

DATASET_ROOT=/shared/nitishg/data/qdmr-processed/QDMR-high-level
DATASET_NAME=DROP/resplits

for attn in false true
do
  for SPLIT_TYPE in full full-20 # full-50 full-40 full-30
  do
    for seed in 1 2 3 4 5
    do
      MODEL_DIR=/shared/nitishg/qdmr/semparse-gen/checkpoints/${DATASET_NAME}/${SPLIT_TYPE}/Seq2Seq-elmo/BS_16/INORDER_true/ATTNLOSS_${attn}/S_${seed}
      # MODEL_DIR=/shared/nitishg/qdmr/semparse-gen/checkpoints/${DATASET_NAME}/${SPLIT_TYPE}/Grammar-elmo/BS_16/ATTNLOSS_${attn}/S_${seed}
      MODEL_TAR=${MODEL_DIR}/model.tar.gz
      PREDICTION_DIR=${MODEL_DIR}/predictions
      mkdir ${PREDICTION_DIR}

      DEV=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/dev.json
      DEV_METRICS=${PREDICTION_DIR}/dev_metrics.json
      allennlp evaluate --output-file ${DEV_METRICS} \
                        --cuda-device ${GPU} \
                        --include-package ${INCLUDE_PACKAGE} \
                        ${MODEL_TAR} ${DEV} &

      HELDOUT_TEST=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/heldout_test.json
      HELDOUT_METRICS=${PREDICTION_DIR}/heldout_test_metrics.json
      allennlp evaluate --output-file ${HELDOUT_METRICS} \
                        --cuda-device ${GPU} \
                        --include-package ${INCLUDE_PACKAGE} \
                        ${MODEL_TAR} ${HELDOUT_TEST} &

      IND_UNBIASED_TEST=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/indomain_unbiased_test.json
      IND_UNBIASED_METRICS=${PREDICTION_DIR}/indomain_unbiased_test_metrics.json
      allennlp evaluate --output-file ${IND_UNBIASED_METRICS} \
                        --cuda-device ${GPU} \
                        --include-package ${INCLUDE_PACKAGE} \
                        ${MODEL_TAR} ${IND_UNBIASED_TEST} &

      IND_SKEWED_TEST=${DATASET_ROOT}/${DATASET_NAME}/${SPLIT_TYPE}/indomain_skewed_test.json
      IND_SKEWED_METRICS=${PREDICTION_DIR}/indomain_skewed_test_metrics.json
      allennlp evaluate --output-file ${IND_SKEWED_METRICS} \
                        --cuda-device ${GPU} \
                        --include-package ${INCLUDE_PACKAGE} \
                        ${MODEL_TAR} ${IND_SKEWED_TEST}
    done
  done
done
