#!/bin/bash

# 设置下载目录
DATASET_DIR=~/datasets/coco_pose
mkdir -p $DATASET_DIR
cd $DATASET_DIR

echo "📥 开始下载 COCO 2017 关键点数据集（用于姿态估计）..."

# 下载 train2017 图像
if [ ! -d "train2017" ]; then
    echo "▶ 下载 train2017 图像..."
    wget -c http://images.cocodataset.org/zips/train2017.zip
    unzip -q train2017.zip
    rm train2017.zip
else
    echo "✅ train2017 已存在，跳过下载"
fi

# 下载 val2017 图像
if [ ! -d "val2017" ]; then
    echo "▶ 下载 val2017 图像..."
    wget -c http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip
    rm val2017.zip
else
    echo "✅ val2017 已存在，跳过下载"
fi

# 下载关键点标注文件
if [ ! -d "annotations" ]; then
    echo "▶ 下载标注文件（包含关键点信息）..."
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
else
    echo "✅ annotations 文件夹已存在，跳过下载"
fi

# 下载 Ultralytics 特定的 labels 文件（YOLO 格式）
if [ ! -d "labels" ]; then
    echo "▶ 下载 YOLO Pose 格式标签（Ultralytics 制作）..."
    curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017_labels-pose.zip -o coco2017_labels-pose.zip
    unzip -q coco2017_labels-pose.zip
    rm coco2017_labels-pose.zip
else
    echo "✅ labels 已存在，跳过下载"
fi

echo "🎉 数据集准备完成！已下载到：$DATASET_DIR"
echo "📂 请检查：train2017, val2017, annotations, labels 是否完整"
