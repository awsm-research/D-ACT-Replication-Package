PROJ_NAME=$1 # android, google, ovirt
TRAIN_STEP=$2


BASE_DATA_DIR=../../dataset/dataset-time-ignore/$PROJ_NAME/
MODEL_DIR=./model_dump/fine-tuning_with_pre-training-time-ignore/$PROJ_NAME/
PYTORCH_DIR=./pytorch_dump-time-ignore/$PROJ_NAME/$TRAIN_STEP
PREDICTION_DIR=./prediction-time-ignore/$PROJ_NAME/final_prediction

python generate_predictions.py 1 $BASE_DATA_DIR $PYTORCH_DIR/pytorch_model.bin $PREDICTION_DIR/ test.tsv

python generate_predictions.py 5 $BASE_DATA_DIR $PYTORCH_DIR/pytorch_model.bin $PREDICTION_DIR/ test.tsv

python generate_predictions.py 10 $BASE_DATA_DIR $PYTORCH_DIR/pytorch_model.bin $PREDICTION_DIR/ test.tsv
