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
IMG_PATH="${BASE_DIR}/rgb"
DEPTH_PATH="${BASE_DIR}/depth"

# echo "Running COLMAP Cover..."
# cd gs/ || exit
# python convert.py --source_path "$BASE_DIR"
# cd ..
# 深度估计
if [ -d "$DEPTH_PATH" ] && [ "$(ls -A $DEPTH_PATH)" ]; then
  echo "Depth already exists in $DEPTH_PATH, skipping depth prediction..."
else
  echo "Running Depth Prediction..."
  cd Depth-Anything-V2/ || exit
  python run.py --encoder vitl --pred-only --grayscale --img-path "$IMG_PATH" --outdir "$DEPTH_PATH"
  cd ..
fi

cd gs
# # 深度缩放
# echo "Scaling Depth..."
# cd gs || exit
# python utils/make_depth_scale.py --base_dir "$BASE_DIR" --depths_dir "$DEPTH_PATH"

echo "Training..."
python train.py -s "$BASE_DIR"  -d "$DEPTH_PATH" --data_device "cpu"\
  --exposure_lr_init 0.001 \
  --exposure_lr_final 0.0001 \
  --exposure_lr_delay_steps 5000 \
  --exposure_lr_delay_mult 0.001 \
  --iterations 30000 \
  --position_lr_max_steps 30000 \
  --densification_interval 30 \
  --opacity_reset_interval 3000 \
  --densify_from_iter 3000 \
  --densify_until_iter 24000 \
  --save_iterations 7000 30000 \
  --test_iterations 7000 30000 \
  --checkpoint_iterations 30000 \
  --sh_degree 3 \
  --resolution 1 \
  --single_read \
  --vggt_test \
  --used_mask \
  --train_test_exp
