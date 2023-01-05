PROJECT=$1 #android,google,ovirt

BASE_DATA_DIR=../../dataset/dataset-time-ignore
MODEL_DIR=fine-tuning_with_pre-training-time-ignore

python train-model.py $PROJECT $BASE_DATA_DIR $MODEL_DIR
