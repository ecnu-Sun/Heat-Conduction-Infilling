import xarray as xr
import numpy as np

# --- 文件路径和变量名 ---
file_a_path = '/hdd/SunZL/data/BCC_nearest/BCC1/thetao_Omon_BCC-CSM2-MR_historical_r1i1p1f1_gn_1850_01.nc'
file_b_path = '/hdd/SunZL/data/ReconMOST-testdata/EN.4.2.2.f.analysis.g10.200011.nc'
var_a_name = 'thetao'
var_b_name = 'temperature'

# --- 使用 xarray 读取数据 ---
# xarray 会自动处理 _FillValue 并统一转换为 np.nan
ds_a = xr.open_dataset(file_a_path)
ds_b = xr.open_dataset(file_b_path)

# 提取变量，并使用 .squeeze() 移除大小为1的维度（如 time）
data_a = ds_a[var_a_name].squeeze()
data_b = ds_b[var_b_name].squeeze()


# --- 比较函数 (现在接收Numpy数组) ---
def compare_missing_numpy(arr_a, arr_b, level_name):
    """使用 Numpy 数组比较缺失值并打印结果"""
    # 现在arr_a和arr_b是纯Numpy数组，可以直接使用np.isnan
    missing_a = np.isnan(arr_a)
    missing_b = np.isnan(arr_b)

    num_missing_a = np.sum(missing_a)
    num_missing_b = np.sum(missing_b)

    # 直接进行布尔运算
    a_missing_b_not = np.sum(missing_a & ~missing_b)
    b_missing_a_not = np.sum(missing_b & ~missing_a)

    print(f"--- {level_name} ---")
    print(f"文件A 缺失值数量: {num_missing_a}")
    print(f"文件B 缺失值数量: {num_missing_b}")
    print(f"A缺失但B有效的格点数: {a_missing_b_not}")
    print(f"B缺失但A有效的格点数: {b_missing_a_not}")
    print("-" * 20)

# --- 进行比较 ---

# 比较前10层
data_a_top10 = data_a.isel(lev=slice(0, 10))
data_b_top10 = data_b.isel(depth=slice(0, 10))

# 【修正处】: 传入 .values 剥离坐标信息，只比较纯数据
compare_missing_numpy(data_a_top10.values, data_b_top10.values, "前10层")


# 比较所有层
# 【修正处】: 传入 .values 剥离坐标信息，只比较纯数据
compare_missing_numpy(data_a.values, data_b.values, "所有层")

# 关闭文件
ds_a.close()
ds_b.close()