
PROJ_NAME=$1
MODEL_TYPE=$2
CKPT=$3
BEAM_SIZE=$4

TEST_INPUT_FILE_PATH='../dataset/final-dataset-no-space-special-chars-latest-version-time-wise/'$PROJ_NAME'/test/initial_ver.txt'
TEST_GT_FILE_PATH='../dataset/final-dataset-no-space-special-chars-latest-version-time-wise/'$PROJ_NAME'/test/approved_ver.txt'

EXP_NAME=$MODEL_TYPE'_without_code_diff_representation_time_wise'


python run-model.py \
    --do_test \
    --test_input_file_path $TEST_INPUT_FILE_PATH \
    --test_gt_file_path $TEST_GT_FILE_PATH \
    --exp_name $EXP_NAME \
    --model_type $MODEL_TYPE \
    --proj_name $PROJ_NAME \
    --selected_train_step $CKPT \
    --beam_size $BEAM_SIZE

