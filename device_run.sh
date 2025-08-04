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
  --iterations 30000\
  --position_lr_max_steps 30000\
  --densification_interval 100\
  --opacity_reset_interval 3000\
  --densify_from_iter 500\
  --densify_until_iter 15000\
  --save_iterations 7000 30000\
  --test_iterations 7000 30000\
  --single_read \
  --used_mask \
  --train_test_exp
