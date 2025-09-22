import numpy as np
import os
import glob

# ==============================================================================
# 1. 请在这里修改您的配置
# ==============================================================================

# 输入: 包含原始NaN值的GT文件夹 (任选一个，train或val都可以)
# !! 注意：是处理前的、含有NaN的文件夹，不是JP1-fillna !!
SRC_GT_WITH_NAN_DIR = "/hdd/SunZL/data/SuperResolutionData/SEA/val_GT"

# 输出: 您希望保存这个唯一掩码文件的位置和文件名
MASK_SAVE_PATH = "/hdd/SunZL/data/SuperResolutionData/SEA/SEA_ocean_mask.npy"

# ==============================================================================
# 2. 脚本主逻辑
# ==============================================================================

def create_static_mask(src_dir, save_path):
    """
    从一个样本文件中创建唯一的静态海洋掩码。
    """
    print(f"--- 正在从源文件夹创建静态掩码 ---")
    print(f"源: {src_dir}")
    print(f"目标: {save_path}")

    # 查找所有.npy文件
    npy_files = glob.glob(os.path.join(src_dir, '*.npy'))
    
    if not npy_files:
        print(f"错误：在目录 {src_dir} 中没有找到任何 .npy 文件。")
        return

    # 随便取第一个文件来创建掩码
    sample_file = npy_files[0]
    print(f"使用样本文件: {os.path.basename(sample_file)}")
    
    try:
        # 加载含有NaN的原始数据
        data_with_nan = np.load(sample_file)
        
        # 创建布尔掩码：非NaN的区域(海洋)为True，NaN区域(陆地)为False
        ocean_mask = ~np.isnan(data_with_nan)
        
        # 保存布尔掩码文件
        np.save(save_path, ocean_mask)
        print(f"\n成功创建并保存掩码文件！")

    except Exception as e:
        print(f"\n创建掩码时发生错误: {e}")

if __name__ == "__main__":
    create_static_mask(SRC_GT_WITH_NAN_DIR, MASK_SAVE_PATH)