#!/usr/bin/env bash

TEST_FILE=$1
MODEL_TAR=$2
OUTPUT_FILE=$3
GPU=$4
INCLUDE_PACKAGE=$5

# BEAMSIZE=$6

allennlp evaluate --output-file ${OUTPUT_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${TEST_FILE}

# --overrides "{"model": {"decoder_beam_search": {"beam_size": 16}}}" \