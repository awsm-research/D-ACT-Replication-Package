PROJECT=android #android, google, ovirt

BASE_DATA_DIR=../../../dataset/dataset-time-wise
SAVE_DIR=../BPE-2000-time-wise-with-code-diff-representation
BIN_DATA_DIR=../binary-data-time-wise-with-code-diff-representation

bash subword_tokenize.sh $BASE_DATA_DIR/$PROJECT $SAVE_DIR/$PROJECT

python generate_binary_data.py $SAVE_DIR/$PROJECT/ $BIN_DATA_DIR/$PROJECT/

