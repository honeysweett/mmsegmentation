#!/bin/bash

# 定义源目录和目标目录变量
SOURCE_DIR="/mmdetection/MTSD_dataset/OpenDataLab___Mapillary_Traffic_Sign_Dataset/train_images"
DEST_DIR="/mmdetection/images_train-test_bstld-mtsd/train"

# 检查目标目录是否存在，如果不存在则创建它
if [ ! -d "$DEST_DIR" ]; then
    mkdir -p "$DEST_DIR"
fi

# 使用cp命令复制文件
mv -v "$SOURCE_DIR"/* "$DEST_DIR"

# 输出完成信息
echo "Files have been copied from $SOURCE_DIR to $DEST_DIR"

# /mmdetection/MTSD_dataset/OpenDataLab___Mapillary_Traffic_Sign_Dataset/raw/mtsd_v2_fully_annotated/splits/train.txt