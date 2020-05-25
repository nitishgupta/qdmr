#!/usr/bin/env

ROOT_DIR=/shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP

#for DATASET in drop-standard-ds drop-standard drop-template-manual drop-template-manual-ds
#do
#  # Convert training data into fast_align input formats for different program representations
#  python -m parse_dataset.fast_align.fastalign_input --qdmr_json ${ROOT_DIR}/${DATASET}/train.json
#  # Run fast-align on different program representations
#  for FILENAME in train.fast_align.seq2seq.inorder train.fast_align.seq2seq train.fast_align.grammar
#  do
#    fast_align -i ${ROOT_DIR}/${DATASET}/${FILENAME} -d -o -v -r > ${ROOT_DIR}/${DATASET}/${FILENAME}.output
#  done
#  # Append QDMRExample with fast-align output for different program representations
#  python -m parse_dataset.fast_align.fastalign_output --qdmr_json ${ROOT_DIR}/${DATASET}/train.json
#
#done

DATASET=drop-programs
python -m parse_dataset.fast_align.prepare_fastalign_format --input_dir ${ROOT_DIR}/${DATASET}

for FILENAME in all_examples.fast_align.seq2seq.inorder all_examples.fast_align.seq2seq all_examples.fast_align.grammar
do
  fast_align -i ${ROOT_DIR}/${DATASET}/${FILENAME} -d -o -v -r > ${ROOT_DIR}/${DATASET}/${FILENAME}.output
done

python -m parse_dataset.fast_align.add_fastalign_to_qdmr --input_dir ${ROOT_DIR}/${DATASET}


