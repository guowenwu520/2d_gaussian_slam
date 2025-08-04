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
IMG_PATH="${BASE_DIR}/input"
DEPTH_PATH="${BASE_DIR}/depth"

echo "Running COLMAP Cover..."
cd gs/ || exit
python convert.py --source_path "$BASE_DIR"
cd ..
# 深度估计
if [ -d "$DEPTH_PATH" ] && [ "$(ls -A $DEPTH_PATH)" ]; then
  echo "Depth already exists in $DEPTH_PATH, skipping depth prediction..."
else
  echo "Running Depth Prediction..."
  cd Depth-Anything-V2/ || exit
  python run.py --encoder vitl --pred-only --grayscale --img-path "$IMG_PATH" --outdir "$DEPTH_PATH"
  cd ..
fi

# 深度缩放
echo "Scaling Depth..."
cd gs || exit
python utils/make_depth_scale.py --base_dir "$BASE_DIR" --depths_dir "$DEPTH_PATH"

# 训练
echo "Training..."
python train.py -s "$BASE_DIR" -d "$DEPTH_PATH" --data_device "cpu"\
  --exposure_lr_init 0.001 \
  --exposure_lr_final 0.0001 \
  --exposure_lr_delay_steps 5000 \
  --exposure_lr_delay_mult 0.001 \
  --single_read \
  --train_test_exp
