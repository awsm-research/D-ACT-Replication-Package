PROJECT=$1 #android,google,ovirt

BASE_DATA_DIR=../../dataset/final-dataset-no-space-special-chars-latest-version-random-split
MODEL_DIR=fine-tuning_with_pre-training-random-split

python train-model.py $PROJECT $BASE_DATA_DIR $MODEL_DIR
