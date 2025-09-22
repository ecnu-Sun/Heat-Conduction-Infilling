import os
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# 0. 新增并行配置
# ==============================================================================
# 设置并行工作的进程数，建议设置为您的CPU核心数或稍小的值
MAX_WORKERS = 32

# ==============================================================================
# 1. 您的核心算法 (保持不变)
# ==============================================================================

def gauss_seidel_sor_vectorized(T, land_mask, omega=1.8, max_iter=5000, tol=1e-4):
    """
    使用向量化操作的SOR迭代法求解二维拉普拉斯方程
    采用红黑排序(Red-Black ordering)来实现并行更新
    """
    T_filled = T.copy()
    ny, nx = T.shape
    
    i_indices, j_indices = np.ogrid[:ny, :nx]
    red_mask = ((i_indices + j_indices) % 2 == 0) & land_mask
    black_mask = ((i_indices + j_indices) % 2 == 1) & land_mask
    
    for iteration in range(max_iter):
        T_old = T_filled.copy()
        for mask in [red_mask, black_mask]:
            if np.sum(mask) == 0:
                continue
            T_left = np.roll(T_filled, 1, axis=1)
            T_right = np.roll(T_filled, -1, axis=1)
            T_up = np.roll(T_filled, 1, axis=0)
            T_down = np.roll(T_filled, -1, axis=0)
            
            T_left[:, 0] = T_filled[:, 0]
            T_right[:, -1] = T_filled[:, -1]
            T_up[0, :] = T_filled[0, :]
            T_down[-1, :] = T_filled[-1, :]
            
            neighbor_sum = T_left + T_right + T_up + T_down
            average = neighbor_sum / 4.0
            T_filled[mask] = T_filled[mask] + omega * (average[mask] - T_filled[mask])
        
        if np.sum(land_mask) > 0:
            max_change = np.max(np.abs(T_filled[land_mask] - T_old[land_mask]))
            if max_change < tol:
                return T_filled
    
    return T_filled

# ==============================================================================
# 2. 路径配置 (保持不变)
# ==============================================================================
SRC_ROOT_DIR = "/hdd/SunZL/data/SuperResolutionData/SEA-fillna"
DEST_ROOT_DIR = "/hdd/SunZL/data/SuperResolutionData/SEA-LAPLACE"
# (后面的路径定义保持不变)
SRC_TRAIN_GT_DIR = os.path.join(SRC_ROOT_DIR, "train_GT")
SRC_TRAIN_LQ_DIR = os.path.join(SRC_ROOT_DIR, "train_LQ")
SRC_VAL_GT_DIR = os.path.join(SRC_ROOT_DIR, "val_GT")
SRC_VAL_LQ_DIR = os.path.join(SRC_ROOT_DIR, "val_LQ")
DEST_TRAIN_GT_DIR = os.path.join(DEST_ROOT_DIR, "train_GT")
DEST_TRAIN_LQ_DIR = os.path.join(DEST_ROOT_DIR, "train_LQ")
DEST_VAL_GT_DIR = os.path.join(DEST_ROOT_DIR, "val_GT")
DEST_VAL_LQ_DIR = os.path.join(DEST_ROOT_DIR, "val_LQ")
PATH_MAPPING = {
    SRC_TRAIN_GT_DIR: DEST_TRAIN_GT_DIR,
    SRC_TRAIN_LQ_DIR: DEST_TRAIN_LQ_DIR,
    SRC_VAL_GT_DIR: DEST_VAL_GT_DIR,
    SRC_VAL_LQ_DIR: DEST_VAL_LQ_DIR,
}

# ==============================================================================
# 3. 脚本主逻辑 (已修改为并行处理)
# ==============================================================================

def process_single_file(file_path, dest_dir):
    """
    这是处理单个文件的“工人”函数，会被并行调用。
    """
    try:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, filename)

        if os.path.exists(dest_path):
            return f"已存在, 跳过: {filename}"
            
        initial_temp = np.load(file_path)
        land_mask = (initial_temp == 0)
        
        if not np.any(land_mask):
            np.save(dest_path, initial_temp)
            return f"无陆地, 复制: {filename}"

        filled_temp = gauss_seidel_sor_vectorized(
            T=initial_temp, 
            land_mask=land_mask, 
            omega=1.85,
            max_iter=5000, 
            tol=1e-3
        )
        np.save(dest_path, filled_temp.astype(np.float32))
        return f"处理完成: {filename}"
    except Exception as e:
        return f"处理失败: {os.path.basename(file_path)} - {e}"

def process_folder_parallel(src_dir, dest_dir):
    """
    使用多进程并行处理文件夹中的所有.npy文件。
    """
    if not os.path.exists(src_dir):
        print(f"!! 警告: 源文件夹不存在，跳过 -> {src_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    npy_files = glob.glob(os.path.join(src_dir, '*.npy'))
    
    if not npy_files:
        print(f"警告：在目录 {src_dir} 中没有找到任何 .npy 文件。")
        return

    print(f"\n--- 正在并行处理 {len(npy_files)} 个文件 (使用 {MAX_WORKERS} 个进程) ---")
    print(f"源: {src_dir}")
    print(f"目标: {dest_dir}")
    
    # 使用ProcessPoolExecutor进行多进程处理
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_file, fp, dest_dir): fp for fp in npy_files}
        
        # 使用tqdm显示处理进度
        for future in tqdm(as_completed(futures), total=len(npy_files), desc="处理进度"):
            # 这里可以获取每个任务的结果，但我们暂时不需要做什么
            result = future.result()
            # if "失败" in result: # 如果需要，可以取消注释来打印失败信息
            #     print(result)

if __name__ == "__main__":
    # 确保脚本在主程序块中运行，这是多进程所必需的
    for src, dest in PATH_MAPPING.items():
        process_folder_parallel(src, dest)

    print("\n==================================================")
    print("所有数据已应用拉普拉斯填充！")
    print(f"处理后的文件保存在: {DEST_ROOT_DIR}")
    print("==================================================")