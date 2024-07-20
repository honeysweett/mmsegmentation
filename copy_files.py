import os
import shutil

def copy_all_files(src_dir, dest_dir):
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 遍历源目录中的所有文件
    for filename in os.listdir(src_dir):
        # 构建源文件路径和目标文件路径
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        # 复制文件
        shutil.copy2(src_path, dest_path)
        print(f"Copied: {filename}")

# 文件路径
src_dir = '/mmdetection/images_train-test_bstld-mtsd/test'
dest_dir = '/share/zhulin/images/test'

copy_all_files(src_dir, dest_dir)
