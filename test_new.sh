#!/bin/bash

# 使用方式: ./test.sh /sobey/new_dir/sobey_office
BASE_DIR=$1

if [ -z "$BASE_DIR" ]; then
  Example "Usage: $0 /sobey/new_dir/sobey_office"
  exit 1
fi


# 训练
cd gs || exit
echo "Training..."
python train.py -s "$BASE_DIR" -d "$DEPTH_PATH" --data_device "cpu"\
  --iterations 30000\
  --exposure_lr_init 0.001 \
  --exposure_lr_final 0.0001 \
  --exposure_lr_delay_steps 5000 \
  --exposure_lr_delay_mult 0.001 \
  --train_test_exp
