PROJECT=$1 #android, google, ovirt

BASE_DATA_DIR=../../../dataset/final-dataset-no-space-special-chars-latest-version-random-split
SAVE_DIR=../BPE-2000-random-split
BIN_DATA_DIR=../binary-data-random-split

bash subword_tokenize.sh $BASE_DATA_DIR/$PROJECT $SAVE_DIR/$PROJECT


BASE_DATA_DIR=../../../dataset/final-dataset-no-space-special-chars-latest-version-time-wise
SAVE_DIR=../BPE-2000-time-wise
BIN_DATA_DIR=../binary-data-time-wise

bash subword_tokenize.sh $BASE_DATA_DIR/$PROJECT $SAVE_DIR/$PROJECT

