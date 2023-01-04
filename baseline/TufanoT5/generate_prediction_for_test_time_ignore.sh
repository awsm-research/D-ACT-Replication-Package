PROJ_NAME=$1 # android, google, ovirt
TRAIN_STEP=$2


BASE_DATA_DIR=../../dataset/final-dataset-no-space-special-chars-latest-version-random-split/$PROJ_NAME/
MODEL_DIR=./model_dump/fine-tuning_with_pre-training-init-diff-app-random-split/$PROJ_NAME/
PYTORCH_DIR=./pytorch_dump-init-diff-app-random-split/$PROJ_NAME/$TRAIN_STEP
PREDICTION_DIR=./prediction-init-diff-app-random-split/$PROJ_NAME/final_prediction

python generate_predictions.py 1 $BASE_DATA_DIR $PYTORCH_DIR/pytorch_model.bin $PREDICTION_DIR/ test-init-diff-app.tsv

python generate_predictions.py 5 $BASE_DATA_DIR $PYTORCH_DIR/pytorch_model.bin $PREDICTION_DIR/ test-init-diff-app.tsv

python generate_predictions.py 10 $BASE_DATA_DIR $PYTORCH_DIR/pytorch_model.bin $PREDICTION_DIR/ test-init-diff-app.tsv
