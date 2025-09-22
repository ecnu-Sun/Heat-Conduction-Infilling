import xarray as xr
import os
import glob
import numpy as np

dstrain=xr.open_dataset('/data/coding/data/ReconMOST-train-FIOESM/thetao_Omon_FIO-ESM-2-0_historical_r1i1p1f1_gn_188001-188912.nc')
dstest=xr.open_dataset('/data/coding/data/ReconMOST-testdata/EN.4.2.2.f.analysis.g10.200001.nc')
# 提取 DataArray 并转换为 NumPy 数组
print(dstrain)
thetao_np = dstrain['thetao'].to_numpy()

# 手动计算并打印统计信息 (使用对 NaN 安全的函数)
print("--- Descriptive Statistics (manual) ---")
print(f"Count: {np.count_nonzero(~np.isnan(thetao_np))}") # 计算非 NaN 值的数量
print(f"Mean: {np.nanmean(thetao_np)}")
print(f"Std Dev: {np.nanstd(thetao_np)}")
print(f"Min: {np.nanmin(thetao_np)}")
print(f"25%: {np.nanpercentile(thetao_np, 25)}")
print(f"50% (Median): {np.nanpercentile(thetao_np, 50)}")
print(f"75%: {np.nanpercentile(thetao_np, 75)}")
print(f"Max: {np.nanmax(thetao_np)}")
print("------------------------------------")
print(dstest)
