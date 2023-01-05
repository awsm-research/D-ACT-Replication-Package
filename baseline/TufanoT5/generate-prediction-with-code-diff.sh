# PROJECT=ovirt #android,google,ovirt
PROJECT=$1
TRAIN_STEP=$2


BASE_DATA_DIR=../../dataset-complete/dataset-time-wise/$PROJECT

EXP_NAME=T5_with_token_level_code_diff_info_time_wise


CUDA_VISIBLE_DEVICES=0 python run-model-with-code-diff.py --proj $PROJECT --test_file_dir $BASE_DATA_DIR/test --do_test --exp_name $EXP_NAME --selected_train_step $TRAIN_STEP --beam_size 1

CUDA_VISIBLE_DEVICES=0 python run-model-with-code-diff.py --proj $PROJECT --test_file_dir $BASE_DATA_DIR/test --do_test --exp_name $EXP_NAME --selected_train_step $TRAIN_STEP --beam_size 5 --eval_batch_size 8

CUDA_VISIBLE_DEVICES=0 python run-model-with-code-diff.py --proj $PROJECT --test_file_dir $BASE_DATA_DIR/test --do_test --exp_name $EXP_NAME --selected_train_step $TRAIN_STEP --beam_size 10 --eval_batch_size 4
