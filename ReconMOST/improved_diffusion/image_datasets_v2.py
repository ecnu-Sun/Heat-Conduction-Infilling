# from PIL import Image
import blobfile as bf
from mpi4py import MPI
import pandas as pd
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import os
import re

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)

    #all_files, train_entries = _list_image_files_recursively(data_dir)
    #原先错用了函数，_list_files_split_train_recursively只能返回一个列表
    all_files, train_entries = _list_files_split_train_recursively(data_dir)
    # print("Load Data from Mode: ", data_dir)
    # print("Choose Train Entries: ", train_entries)
    for mode in train_entries:
        print("Load Data from Mode: ", mode)
    classes = None
    # 在reconMOST中，一直让class_cond为false即可，因为不涉及类别，或者干脆注释掉
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)  
        ext = entry.split(".")[-1]
        # add NPY file.
        # if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy", "nc"]:
        if "." in entry and ext.lower() in ['nc']:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

# use in single mode train(split every mode into train and test)
def _list_files_split_train_recursively(data_dir):
    results = []
    train_entries = []
    for entry in sorted(bf.listdir(data_dir))[0:-1]:
        train_entries.append(entry)
        full_path = bf.join(data_dir, entry)
        results.extend(_list_image_files_recursively(full_path))
    return results, train_entries


# use in multi mode train(ablation 2)
def _list_multi_mode_train_recursively(data_dir):
    results = []
    train_entries = []
    modes = ['FIO-ESM-2-0','BCC-CSM2-MR','MRI-ESM2-0','CanESM5','IPSL-CM6A-LR','FGOALS-g3','FGOALS-f3-L']
    for mode in modes:
        full_path = bf.join(data_dir, mode)
        train_entries.append(full_path)
        results.extend(_list_image_files_recursively(full_path))
    return results, train_entries


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution  # image_size 180*360
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        
        #以下代码用于nino指数嵌入
        self.nino_dict={}
        df=pd.read_csv("/hdd/SunZL/data/nino34.long.anom_previous.csv")
        df.columns=df.columns.str.strip()
        for _,row in df.iterrows():
            date=row['Date']
            nino_value=row['NINA34']
            year=int(date[:4])
            month = int(date[5:7])
            self.nino_dict[(year, month)] = nino_value
        #以上代码用于nino指数嵌入

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        # singlelayer.npy
        # reconMOST用不到
        if path.endswith(".npy"):
            with bf.BlobFile(path, "rb") as f:
                # pil_image = Image.open(f)
                # pil_image.load()
                arr = np.load(f) # array [180, 360]
                arr = np.nan_to_num(arr, nan=0.0)
                arr = 2 * (arr + 5) / 45 - 1   # rescale [-1, 1]
                arr = arr.astype(np.float32)

        # multi-layer.nc
        elif path.endswith(".nc"):
            ds = xr.open_dataset(path)
            arr = ds.thetao.values  # 42, 173*360, -83-89
            arr = np.nan_to_num(arr, nan=0.0)
            arr = 2 * (arr + 5) / 45 - 1   # [-5, 40] rescale to [-1, 1]
            arr = arr.astype(np.float32)
            
        out_dict = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
        #以下代码用于时间条件版本
        basename = os.path.basename(path)
        year,month=None,None
        # 模式1: 尝试匹配 YYYY_MM.nc 格式
        match = re.search(r'(\d{4})_(\d{2})', basename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
        else:
            # 模式2: 尝试匹配 YYYYMM.nc 格式
            match = re.search(r'(\d{6})', basename)
            if match:
                date_str = match.group(1)
                year = int(date_str[:4])
                month = int(date_str[4:])
        # print(year)
        # print(month)
        out_dict["year"] = th.tensor(year, dtype=th.int32)
        out_dict["month"] = th.tensor(month, dtype=th.int32)
        #以上代码用于时间条件版本
        ##以下代码获取nino
        nino = self.nino_dict.get((year, month), 0.0)
        out_dict["nino"] = th.tensor(nino, dtype=th.float32)
        ##以上代码获取nino
        

        if len(arr.shape) == 2:
            # reshape to CHW
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        if len(arr.shape) == 4:
            # reshape to CHW
            arr = arr.reshape(arr.shape[1], arr.shape[2], arr.shape[3])
        return arr, out_dict
