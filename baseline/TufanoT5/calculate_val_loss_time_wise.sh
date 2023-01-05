PROJ_NAME=$1 # android, google, ovirt

BASE_DATA_DIR=../../dataset-complete/dataset-time-wise/$PROJ_NAME/
MODEL_DIR=./model_dump/fine-tuning_with_pre-training-time-wise/$PROJ_NAME/
PYTORCH_DIR=./pytorch_dump-time-wise/$PROJ_NAME/
PREDICTION_DIR=./prediction-time-wise/$PROJ_NAME/final_prediction


python calculate_val_loss.py 1 $BASE_DATA_DIR $PYTORCH_DIR/ $PREDICTION_DIR/ eval.tsv
