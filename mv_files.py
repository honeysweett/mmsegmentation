import os
import shutil

def move_all_files(src_dir, dest_dir):
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 遍历源目录中的所有文件
    for filename in os.listdir(src_dir):
        # 构建源文件路径和目标文件路径
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        # 移动文件
        shutil.move(src_path, dest_path)
        print(f"Moved: {filename}")

# 文件路径
src_dir = '/mmdetection/BSTLD_dataset/dataset_train_rgb/train_images'
dest_dir = '/mmdetection/images_train-test_bstld-mtsd/train'

move_all_files(src_dir, dest_dir)
