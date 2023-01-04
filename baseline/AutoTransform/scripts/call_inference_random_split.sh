PROJECT=$1 #android, google, ovirt
TRAIN_STEP=$2

BASE_DATA_DIR= ../../../dataset/final-dataset-no-space-special-chars-latest-version-random-split


SAVE_DIR=../BPE-2000-random-split
BIN_DATA_DIR=../binary-data-random-split
MODEL_DIR=../model-random-split/$PROJECT
PREDICTION_DIR=../prediction-random-split/$PROJECT/test


bash inference.sh $BIN_DATA_DIR/$PROJECT/ $MODEL_DIR  $SAVE_DIR/$PROJECT/test/initial_ver_init_diff_app.txt $TRAIN_STEP 1 $PREDICTION_DIR/

bash inference.sh $BIN_DATA_DIR/$PROJECT/ $MODEL_DIR  $SAVE_DIR/$PROJECT/test/initial_ver_init_diff_app.txt $TRAIN_STEP 5 $PREDICTION_DIR/

bash inference.sh $BIN_DATA_DIR/$PROJECT/ $MODEL_DIR  $SAVE_DIR/$PROJECT/test/initial_ver_init_diff_app.txt $TRAIN_STEP 10 $PREDICTION_DIR/
