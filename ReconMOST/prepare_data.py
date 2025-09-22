# import xarray as xr
# import numpy as np
# import os
# import xesmf as xe
# from joblib import Parallel, delayed
# import multiprocessing
# import glob

# def split(input_file):
#     ds = xr.open_dataset(input_file)
#     time_var = ds.time
#     grouped = ds.groupby(time_var.dt.strftime('%Y-%m'))
    
#     # 返回数据列表而不是文件列表
#     data_list = []
#     for group_name, group_data in grouped:
#         year, month = group_name.split('-')
#         filename = f"thetao_Omon_FIO-ESM-2-0_historical_r1i1p1f1_gn_{year}_{month}.nc"
#         print(f"Processing {filename}")
#         data_list.append((filename, group_data))
    
#     ds.close()
#     return data_list

# def interp_lonlat_lev(source_ds, target_ds, name="thetao", save_path="/data/coding/data/no_aligned_FIO/FIO"):
    
#     source_ds = source_ds.rename({'longitude': 'lon', 'latitude': 'lat'})
#     source_ds['lat'] = source_ds['lat'].sel(drop=True)
#     source_ds['lon'] = source_ds['lon'].sel(drop=True)
    
#     regridder = xe.Regridder(source_ds, target_ds, 'bilinear')
#     regrided = regridder(source_ds.thetao)
#     regrided_ds = regrided.to_dataset(name="thetao")
    
#     regrided_ds = regrided_ds.rename({'lev': 'depth'})
#     ds_interp = regrided_ds.interp(depth=target_ds.depth)
#     ds_interp = ds_interp.rename({'depth': 'lev'})
    
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
    
#     ds_interp.to_netcdf(f"{save_path}/{name}")
#     print(f"Interpolation complete. Saved to {name}")

# # 修改：处理数据而不是文件
# def process_data(data_tuple, target_ds):
#     filename, source_ds = data_tuple
#     interp_lonlat_lev(source_ds, target_ds, name=filename, save_path="/data/coding/data/ReconMOST_train_processed_en4grid/fgoals-f3-l")

# # 修改:处理所有输入文件
# input_folder = "/data/coding/data/ReconMOST-train-FIOESM/"
# input_files = glob.glob(os.path.join(input_folder, "*.nc"))
# print(f"Found {len(input_files)} input files")

# target_ds = xr.open_dataset('/data/coding/data/ReconMOST-testdata/EN.4.2.2.f.analysis.g10.200001.nc')

# all_data_tuples = []
# for input_file in input_files:
#     print(f"Processing {input_file}")
#     data_tuples = split(input_file)
#     all_data_tuples.extend(data_tuples)

# # num_cores = max(1, int(multiprocessing.cpu_count() * 1))
# # print(f"Using {num_cores} cores for parallel processing")

# Parallel(n_jobs=1)(
#     delayed(process_data)(data_tuple, target_ds) 
#     for data_tuple in all_data_tuples
# )
import xarray as xr
import numpy as np
import os
# os.environ["ESMFMKFILE"]="/home/zju/anaconda3/envs/reconpreparedata/lib/esmf.mk"
import xesmf as xe
from joblib import Parallel, delayed
import multiprocessing
import glob
import fcntl
import time

def split(input_file):
    ds = xr.open_dataset(input_file)
    time_var = ds.time
    grouped = ds.groupby(time_var.dt.strftime('%Y-%m'))
    
    # 返回数据列表而不是文件列表
    data_list = []
    for group_name, group_data in grouped:
        year, month = group_name.split('-')
        filename = f"thetao_Omon_MRI-ESM2-0_historical_r1i1p1f1_gn_{year}_{month}.nc"
        print(f"Processing {filename}")
        data_list.append((filename, group_data))
    
    ds.close()
    return data_list

def safe_write_netcdf(ds, filepath, max_retries=5):
    """安全地写入NetCDF文件，带重试机制"""
    for attempt in range(max_retries):
        try:
            ds.to_netcdf(filepath)
            return
        except RuntimeError as e:
            if "HDF error" in str(e) and attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # 递增等待时间
                continue
            else:
                raise

def interp_lonlat_lev(source_ds, target_ds, save_path,name="thetao"):
    
    # source_ds['thetao'] = source_ds['thetao'].fillna(0)
    # source_ds = source_ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    # source_ds['lat'] = source_ds['lat'].sel(drop=True)
    # source_ds['lon'] = source_ds['lon'].sel(drop=True)
    
    # regridder = xe.Regridder(source_ds, target_ds, 'nearest_s2d',periodic=True)
    source_ds = source_ds.rename({
    # 'lon': 'x',               # 1. 将一维坐标变量 'lon' 重命名为 'x'
    # 'lat': 'y',               # 2. 将一维坐标变量 'lat' 重命名为 'y'
    'longitude': 'lon',       # 3. 将二维地理坐标 'longitude' 重命名为 'lon'
    'latitude': 'lat'         # 4. 将二维地理坐标 'latitude' 重命名为 'lat'
})
    regridder = xe.Regridder(source_ds, target_ds, 'bilinear',periodic=True)
    regrided = regridder(source_ds.thetao)
    regrided_ds = regrided.to_dataset(name="thetao")
    
    regrided_ds = regrided_ds.rename({'lev': 'depth'})
    ds_interp = regrided_ds.interp(depth=target_ds.depth, method='linear')
    ds_interp = ds_interp.rename({'depth': 'lev'})
    
    if not os.path.exists(save_path):
        os.makedirs(save_path,exist_ok=True)
    
    filepath = f"{save_path}/{name}"
    safe_write_netcdf(ds_interp, filepath)
    print(f"Interpolation complete. Saved to {name}")

# 修改：处理数据而不是文件
def process_data(data_tuple, target_ds):
    filename, source_ds = data_tuple
    interp_lonlat_lev(source_ds, target_ds, name=filename, save_path="/hdd/SunZL/data/MRI12_regrided/MRI1_regrided")

# 修改:处理所有输入文件
input_folder = "/hdd/SunZL/data/MRI1_RAW"
input_files = glob.glob(os.path.join(input_folder, "*.nc"))
print(f"Found {len(input_files)} input files")

target_ds = xr.open_dataset('/hdd/SunZL/data/ReconMOST-testdata/EN.4.2.2.f.analysis.g10.200001.nc')

all_data_tuples = []
for input_file in input_files:
    print(f"Processing {input_file}")
    data_tuples = split(input_file)
    all_data_tuples.extend(data_tuples)

num_cores = max(1, int(multiprocessing.cpu_count() * 1))
print(f"Using {num_cores} cores for parallel processing")

# 使用较低的并行度，避免HDF5冲突
Parallel(n_jobs= 4)(  # 限制最多4个并行进程
    delayed(process_data)(data_tuple, target_ds)
    for data_tuple in all_data_tuples
)