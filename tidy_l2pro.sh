#!/bin/bash

# 检查参数
if [ -z "$1" ]; then
  echo "Usage: $0 <base_perspective_dir>"
  echo "Example: $0 /sobey/job_dir/perspective"
  exit 1
fi

BASE_DIR="$1"
TIDY_DIR="${BASE_DIR}_tidy"


# 创建输出目录
mkdir -p "$TIDY_DIR/images"
mkdir -p "$TIDY_DIR/input"
mkdir -p "$TIDY_DIR/masks"
mkdir -p "$TIDY_DIR/sparse/0"

# 拷贝图像
cp -r "$BASE_DIR/images/"* "$TIDY_DIR/images"
cp -r "$BASE_DIR/masks/"* "$TIDY_DIR/masks"
cp -r "$BASE_DIR/images/$PERSPECTIVE_NAME/L2PRO/camera_0/"* "$TIDY_DIR/input"
cp -r "$BASE_DIR/sparse/"* "$TIDY_DIR/sparse/0"
