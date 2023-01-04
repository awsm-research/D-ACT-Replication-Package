#!/bin/bash

# DATA_DIR: the directory path to the dataset of the original code of changed methods
# OUTPUT_DIR: the directory for storing the subword tokenized data with the same structure as above.
# NUM_MERGE: the number of merge operations to generate and apply

DATA_DIR=$1 # ../../../dataset/has-space-of-special-chars/$proj/$size
OUTPUT_DIR=$2 #../BPE-2000/$proj/$size
NUM_MERGE=2000

#Setup environment variables
TRAIN_BEFORE=$DATA_DIR"/train/initial_ver_code_diff_info.txt"
TRAIN_AFTER=$DATA_DIR"/train/approved_ver.txt"

echo $TRAIN_BEFORE

mkdir -p $OUTPUT_DIR

#STEP1: Generate merge operations based on the training dataset
subword-nmt learn-joint-bpe-and-vocab --input $TRAIN_BEFORE $TRAIN_AFTER -s $NUM_MERGE -o $OUTPUT_DIR"/train.all.codes" --write-vocabulary $OUTPUT_DIR"/train.code_before.vocab" $OUTPUT_DIR"/train.code_bafter.vocab"

#STEP2: Apply merge operations generated from learn bpe to the train, eval, test and before and after versions. Note that Java keywords listd in Java_keywords.txt will not be subtokenized.
java_keywords=$(cat ./Java_keywords.txt)

mkdir -p $OUTPUT_DIR"/train"
mkdir -p $OUTPUT_DIR"/eval"
mkdir -p $OUTPUT_DIR"/test"
files=("train/initial_ver_code_diff_info.txt" "train/approved_ver.txt" "eval/initial_ver_code_diff_info.txt" "eval/approved_ver.txt" "test/initial_ver_code_diff_info.txt")

for file in ${files[*]}
do
    subword-nmt apply-bpe -c $OUTPUT_DIR"/train.all.codes" --merges $NUM_MERGE -i $DATA_DIR"/$file" -o $OUTPUT_DIR"/$file" --glossaries $java_keywords
done
