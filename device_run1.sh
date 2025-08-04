#!/bin/bash

# 使用方式: ./run_pipeline.sh hotel
SCENE=$1

if [ -z "$SCENE" ]; then
  echo "Usage: $0 <scene_name>"
  exit 1
fi

# 场景路径设置
# BASE_DIR="/home/ubuntu/workspace/gs/input/${SCENE}"
BASE_DIR=$SCENE
cd gs
echo "Training..."
python train.py -s "$BASE_DIR"  --data_device "cpu"\
  --exposure_lr_init 0.001 \
  --exposure_lr_final 0.0001 \
  --exposure_lr_delay_steps 5000 \
  --exposure_lr_delay_mult 0.001 \
  --position_lr_init 0.00005 \
  --scaling_lr 0.002 \
  --position_lr_final 0.000005 \
  --used_mask \
  --single_read \
  --train_test_exp
