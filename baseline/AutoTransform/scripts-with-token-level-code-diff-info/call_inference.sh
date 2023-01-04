PROJECT=$1 #android, google, ovirt
TRAIN_STEP=$2

SAVE_DIR=../BPE-2000-time-wise-with-code-diff-representation
BIN_DATA_DIR=../binary-data-time-wise-with-code-diff-representation
MODEL_DIR=../model-time-wise-with-token-level-code-diff-info/$PROJECT
PREDICTION_DIR=../prediction-time-wise-with-token-level-code-diff-info/$PROJECT/test


bash inference.sh $BIN_DATA_DIR/$PROJECT/ $MODEL_DIR  $SAVE_DIR/$PROJECT/test/initial_ver_code_diff_info.txt $TRAIN_STEP 1 $PREDICTION_DIR/

bash inference.sh $BIN_DATA_DIR/$PROJECT/ $MODEL_DIR  $SAVE_DIR/$PROJECT/test/initial_ver_code_diff_info.txt $TRAIN_STEP 5 $PREDICTION_DIR/

bash inference.sh $BIN_DATA_DIR/$PROJECT/ $MODEL_DIR  $SAVE_DIR/$PROJECT/test/initial_ver_code_diff_info.txt $TRAIN_STEP 10 $PREDICTION_DIR/
