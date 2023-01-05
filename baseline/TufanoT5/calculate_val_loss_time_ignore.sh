PROJ_NAME=$1 # android, google, ovirt

BASE_DATA_DIR=../../dataset/dataset-time-ignore/$PROJ_NAME/
MODEL_DIR=./model_dump/fine-tuning_with_pre-training-time-ignore/$PROJ_NAME/
PYTORCH_DIR=./pytorch_dump-time-ignore/$PROJ_NAME/
PREDICTION_DIR=./prediction-time-ignore/$PROJ_NAME/

python calculate_val_loss.py 1 $BASE_DATA_DIR $PYTORCH_DIR/ $PREDICTION_DIR/ eval-with-special-tokens.tsv
