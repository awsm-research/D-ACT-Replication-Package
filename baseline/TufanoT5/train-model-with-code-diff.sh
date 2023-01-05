PROJECT=$1

BASE_DATA_DIR=../../dataset-complete/dataset-time-wise/$PROJECT

EXP_NAME=T5_with_token_level_code_diff_info_time_wise


CUDA_VISIBLE_DEVICES=0 python run-model-from-pytorch.py --proj $PROJECT --train_file_dir $BASE_DATA_DIR/train/ --eval_file_dir $BASE_DATA_DIR/eval/ --do_train  --exp_name $EXP_NAME
