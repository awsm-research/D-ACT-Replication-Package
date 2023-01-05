PROJECT=$1 #android,google,ovirt

BASE_DATA_DIR=../../dataset/final-dataset-no-space-special-chars-latest-version-time-wise
MODEL_DIR=fine-tuning_with_pre-training-time-wise

python train-model.py $PROJECT $BASE_DATA_DIR $MODEL_DIR
