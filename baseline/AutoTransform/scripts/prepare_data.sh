PROJECT=$1 #android, google, ovirt

BASE_DATA_DIR=../../../dataset/dataset-time-ignore
SAVE_DIR=../BPE-2000-time-ignore
BIN_DATA_DIR=../binary-data-time-ignore

bash subword_tokenize.sh $BASE_DATA_DIR/$PROJECT $SAVE_DIR/$PROJECT


BASE_DATA_DIR=../../../dataset/dataset-time-wise
SAVE_DIR=../BPE-2000-time-wise
BIN_DATA_DIR=../binary-data-time-wise

bash subword_tokenize.sh $BASE_DATA_DIR/$PROJECT $SAVE_DIR/$PROJECT

