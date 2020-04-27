#!/usr/bin/env

SERIALIZATION_DIR=./resources/semqa/checkpoints/drop/merged_data/date_yd_num_hmyw_cnt_whoarg_0_full/drop_parser_bert/EXCLOSS_false/MMLLOSS_true/aux_true/SUPEPOCHS_0/S_1/BeamSize2_HEM0_wSpanAns
WEIGHTS_TH=best.th

MODEL_ARCHIVE=${SERIALIZATION_DIR}/model.tar.gz

if [ -f "${MODEL_ARCHIVE}" ]; then
    echo "Model archive exists: ${MODEL_ARCHIVE}"
    exit 1
fi

echo "Making model.tar.gz in ${MODEL_ARCHIVE}"

cd ${SERIALIZATION_DIR}

if [ -d "modeltar" ]; then
    echo "modeltar directory exists. Deleting ... "
    rm -r modeltar
fi


mkdir modeltar
cp -r vocabulary modeltar/
cp ${WEIGHTS_TH} modeltar/weights.th
cp config.json modeltar/
cd modeltar
tar -czvf model.tar.gz ./*
cp model.tar.gz ../
cd ../
rm -r modeltar