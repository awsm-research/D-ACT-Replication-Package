
PROJ_NAME=$1
MODEL_TYPE=$2

TRAIN_INPUT_FILE_PATH='../dataset/final-dataset-no-space-special-chars-latest-version-time-wise/'$PROJ_NAME'/train/initial_ver.txt'
TRAIN_GT_FILE_PATH='../dataset/final-dataset-no-space-special-chars-latest-version-time-wise/'$PROJ_NAME'/train/approved_ver.txt'
EVAL_INPUT_FILE_PATH='../dataset/final-dataset-no-space-special-chars-latest-version-time-wise/'$PROJ_NAME'/eval/initial_ver.txt'
EVAL_GT_FILE_PATH='../dataset/final-dataset-no-space-special-chars-latest-version-time-wise/'$PROJ_NAME'/eval/approved_ver.txt'

EXP_NAME=$MODEL_TYPE'_with_code_diff_representation_time_wise'

python run-model.py \
    --do_train \
    --train_input_file_path $TRAIN_INPUT_FILE_PATH \
    --train_gt_file_path $TRAIN_GT_FILE_PATH \
    --eval_input_file_path $EVAL_INPUT_FILE_PATH$ \
    --eval_gt_file_path $EVAL_GT_FILE_PATH$ \
    --exp_name $EXP_NAME \
    --model_type $MODEL_TYPE \
    --proj_name $PROJ_NAME

