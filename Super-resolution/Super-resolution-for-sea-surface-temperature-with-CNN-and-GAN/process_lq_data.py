import os
import numpy as np
import glob
from tqdm import tqdm

# ==============================================================================
# 1. 请在这里修改您的配置
# ==============================================================================

# [输入] 包含原始LQ .npy文件的训练集文件夹
SRC_TRAIN_DIR = "/hdd/SunZL/data/SuperResolutionData/LQ_Dataset_20CRv3/train_LQ"

# [输入] 包含原始LQ .npy文件的测试集/验证集文件夹
SRC_TEST_DIR = "/hdd/SunZL/data/SuperResolutionData/LQ_Dataset_20CRv3/test_LQ"

# [输出] 保存裁剪后训练集LQ文件的文件夹
DEST_TRAIN_DIR = "/hdd/SunZL/data/SuperResolutionData/SEA/train_LQ"

# [输出] 保存裁剪后验证集LQ文件的文件夹
DEST_VAL_DIR = "/hdd/SunZL/data/SuperResolutionData/SEA/val_LQ"

# # ==============================================================================
# # 2. JP1区域的数组索引 (基于1度全球网格计算)
# # ==============================================================================
# LAT_START_INDEX = 120  # 对应 30°N
# LAT_END_INDEX = 136    # 对应 46°N (不含此索引, Python切片特性)

# LON_START_INDEX = 130 # 对应 130°E
# LON_END_INDEX = 146   # 对应 146°E (不含此索引, Python切片特性)
# ==============================================================================
# 2. 东南亚区域的数组索引 (基于1度全球网格计算)
#    (输出 64x64 区域)
# ==============================================================================
# 纬度范围: -22°S 到 42°N
LAT_START_INDEX = 68   # 对应 -22°S (索引 = -22 - (-90))
LAT_END_INDEX = 132    # 对应 42°N (索引 = 42 - (-90))，切片范围共64行

# 经度范围: 95°E 到 159°E
LON_START_INDEX = 95   # 对应 95°E
LON_END_INDEX = 159    # 对应 159°E，切片范围共64列
# ==============================================================================
# 3. 脚本主逻辑 (通常无需修改)
# ==============================================================================

def process_folder(src_dir, dest_dir):
    """
    处理单个文件夹：读取所有.npy，裁剪，然后保存到目标文件夹。
    """
    print(f"\n--- 正在处理源文件夹: {src_dir} ---")
    
    # 确保目标文件夹存在
    os.makedirs(dest_dir, exist_ok=True)
    
    # 查找所有.npy文件
    npy_files = glob.glob(os.path.join(src_dir, '*.npy'))
    
    if not npy_files:
        print(f"警告：在目录 {src_dir} 中没有找到任何 .npy 文件。")
        return

    print(f"找到 {len(npy_files)} 个文件，开始裁剪...")

    for file_path in tqdm(npy_files, desc=f"处理到 -> {os.path.basename(dest_dir)}"):
        # 构建目标文件路径
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, filename)

        if os.path.exists(dest_path):
            continue
            
        try:
            # 加载完整的全球或大区域数据
            global_data = np.load(file_path)
            
            # 使用计算好的索引进行裁剪
            jp1_data = global_data[LAT_START_INDEX:LAT_END_INDEX, LON_START_INDEX:LON_END_INDEX].squeeze()
            
            # 验证裁剪后的尺寸
            if jp1_data.shape != (64, 64):
                print(f"\n警告：文件 {filename} 裁剪后尺寸为 {jp1_data.shape}，非预期。请检查原始文件尺寸或索引。")
                continue
            
            np.save(dest_path, jp1_data)

        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误: {e}")

if __name__ == "__main__":
    # 处理训练集
    process_folder(SRC_TRAIN_DIR, DEST_TRAIN_DIR)
    
    # 处理测试集/验证集
    process_folder(SRC_TEST_DIR, DEST_VAL_DIR)

    print("\n所有LQ数据已按要求裁剪并分类完毕！")
    print(f"训练集LQ数据位于: {DEST_TRAIN_DIR}")
    print(f"验证集LQ数据位于: {DEST_VAL_DIR}")