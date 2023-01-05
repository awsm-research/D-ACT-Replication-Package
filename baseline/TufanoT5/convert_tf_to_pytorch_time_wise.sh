PROJ_NAME=$1 # android, google, ovirt

MODEL_DIR=./model_dump/fine-tuning_with_pre-training-time-wise/$PROJ_NAME/
PYTORCH_DIR=./pytorch_dump-time-wise/$PROJ_NAME/


for ((step=202000;step<=500000;step+=2000)); do
    python tf_2_pytorch_T5.py --tf_checkpoint_path $MODEL_DIR/model.ckpt-$step --config_file config.json --pytorch_dump_path $PYTORCH_DIR/$step
    
done
