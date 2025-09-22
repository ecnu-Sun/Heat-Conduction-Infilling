import xarray as xr
import numpy as np
import os
import xesmf as xe
from joblib import Parallel, delayed
import multiprocessing
import glob
def split(input_file, output_dir="./temp_monthly_files"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ds = xr.open_dataset(input_file)
    time_var = ds.time
    grouped = ds.groupby(time_var.dt.strftime('%Y-%m'))
    for group_name, group_data in grouped:
        year, month = group_name.split('-')
        output_file = os.path.join(output_dir, f"thetao_Omon_FIO-ESM-2-0_historical_r1i1p1f1_gn_{year}_{month}.nc")
        print(f"Saving {output_file}")
        group_data.to_netcdf(output_file)
    file_list = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.nc')]
    ds.close()
    return file_list


def interp_lonlat_lev(source_ds, target_ds, name="thetao", save_path="/data/coding/data/no_aligned_FIO/FIO"):

    # source_ds = source_ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    source_ds['lat'] = source_ds['lat'].sel(drop=True)
    source_ds['lon'] = source_ds['lon'].sel(drop=True)

    regridder = xe.Regridder(source_ds, target_ds, 'bilinear')
    regrided = regridder(source_ds.thetao)
    regrided_ds = regrided.to_dataset(name="thetao")
    
    regrided_ds = regrided_ds.rename({'lev': 'depth'})
    ds_interp = regrided_ds.interp(depth=target_ds.depth)
    ds_interp = ds_interp.rename({'depth': 'lev'})
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    ds_interp.to_netcdf(f"{save_path}/{name}")
    print(f"Interpolation complete. Saved to {name}")


# input_file = "/data/coding/data/ReconMOST-train-FIOESM/thetao_Omon_FIO-ESM-2-0_historical_r1i1p1f1_gn_186001-186912.nc"
# target_ds = xr.open_dataset('/data/coding/data/ReconMOST-testdata/EN.4.2.2.f.analysis.g10.200001.nc')

# file_names = split(input_file)

# def process_file(file_name, target_ds):
#     source_ds = xr.open_dataset(file_name)
#     interp_lonlat_lev(source_ds, target_ds, name=os.path.basename(file_name), save_path="/data/coding/data/ReconMOST_train_processed_en4grid/fgoals-f3-l")
#     source_ds.close()
# 修改：处理所有输入文件
input_folder = "/data/coding/data/ReconMOST-train-FIOESM/"
input_files = glob.glob(os.path.join(input_folder, "*.nc"))
print(f"Found {len(input_files)} input files")

target_ds = xr.open_dataset('/data/coding/data/ReconMOST-testdata/EN.4.2.2.f.analysis.g10.200001.nc')

all_file_names = []
for input_file in input_files:
    print(f"Processing {input_file}")
    file_names = split(input_file)
    all_file_names.extend(file_names)

def process_file(file_name, target_ds):
    source_ds = xr.open_dataset(file_name)
    interp_lonlat_lev(source_ds, target_ds, name=os.path.basename(file_name), save_path="/data/coding/data/ReconMOST_train_processed_en4grid/fgoals-f3-l")
    source_ds.close()

num_cores = max(1, int(multiprocessing.cpu_count() * 1))
print(f"Using {num_cores} cores for parallel processing")

Parallel(n_jobs=num_cores/2)(
    delayed(process_file)(file_name, target_ds) 
    for file_name in all_file_names
)