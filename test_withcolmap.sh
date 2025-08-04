#!/bin/bash

# 使用方式: ./test.sh /sobey/new_dir/sobey_office
BASE_DIR=$1

if [ -z "$BASE_DIR" ]; then
  Example "Usage: $0 /sobey/new_dir/sobey_office"
  exit 1
fi


# 场景路径设置
IMG_PATH="${BASE_DIR}/input"
DEPTH_PATH="${BASE_DIR}/depth"


# colmap整理
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
  --iterations 60000\
  --exposure_lr_init 0.001 \
  --exposure_lr_final 0.0001 \
  --exposure_lr_delay_steps 5000 \
  --exposure_lr_delay_mult 0.001 \
  --train_test_exp
