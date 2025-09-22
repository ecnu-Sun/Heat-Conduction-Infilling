# import os
# import glob
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# from typing import List, Tuple
# from utils import load_nc_file, normalize_temperature


# class SeaTemperatureDataset(Dataset):
#     def __init__(self, data_dir: str, temp_min: float = -5.0, temp_range: float = 45.0):
#         """
#         海温数据集类 - 支持EN4数据格式
        
#         Args:
#             data_dir: 包含.nc文件的目录路径
#             temp_min: 温度最小值
#             temp_range: 温度范围
#         """
#         self.data_dir = data_dir
#         self.temp_min = temp_min
#         self.temp_range = temp_range
        
#         # 递归查找所有.nc文件
#         self.file_paths = []
#         for root, dirs, files in os.walk(data_dir):
#             for file in files:
#                 if file.endswith('.nc'):
#                     self.file_paths.append(os.path.join(root, file))
        
#         self.file_paths.sort()
#         print(f"Found {len(self.file_paths)} .nc files in {data_dir}")
        
#         if len(self.file_paths) == 0:
#             raise ValueError(f"No .nc files found in {data_dir}")
    
#     def __len__(self) -> int:
#         return len(self.file_paths)
    
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
#         """
#         返回归一化的海温数据和文件路径
        
#         Returns:
#             data: 形状为[42, 173, 360]的归一化海温张量（归一化到[-1, 1]）
#             filepath: 源文件路径
#         """
#         filepath = self.file_paths[idx]
        
#         try:
#             # 加载数据
#             data = load_nc_file(filepath)
            
#             # 确保数据形状正确
#             expected_shapes = [(42, 173, 360), (42, 173, 180)] 
            
#             if data.shape not in expected_shapes:
#                 # 如果形状不对，尝试调整
#                 if len(data.shape) == 4 and data.shape[0] == 1:
#                     data = data[0]  # 移除时间维度
#                     print(f"Adjusted data shape from {data.shape} to {data.shape}")
#                 elif len(data.shape) == 3 and data.shape[0] != 42:
#                     # 可能需要转置
#                     data = np.transpose(data, (2, 0, 1))  # 假设深度在最后一维
#                 else:
#                     print(f"Warning: Unexpected data shape {data.shape} in {filepath}")
            
            
#             # 确保最终形状是[42, 173, 360]
#             if data.shape != (42, 173, 360):
#                 print(f"Error: Cannot process data shape {data.shape} from {filepath}")
#                 exit(1)
            
#             # 转换为tensor
#             data = torch.from_numpy(data).float()
            
#             # 处理缺失值（NaN）- 使用温度最小值填充
#             data = torch.nan_to_num(data, nan=self.temp_min)
            
#             # 归一化到[-1, 1]
#             data = normalize_temperature(data, self.temp_min, self.temp_range)
            
#             # 确保数据在[-1, 1]范围内
#             data = torch.clamp(data, -1.0, 1.0)
            
#             return data, filepath
            
#         except Exception as e:
#             print(f"Error loading {filepath}: {e}")
#             # 返回一个零张量作为fallback
#             return torch.zeros(42, 173, 360), filepath


# def create_dataloader(data_dir: str, batch_size: int, num_workers: int = 4,
#                       shuffle: bool = True, **kwargs):
#     """
#     创建数据加载器
#     """
#     dataset = SeaTemperatureDataset(data_dir, **kwargs)
    
#     # 如果数据集为空，返回None
#     if len(dataset) == 0:
#         return None
    
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False  # 测试时不需要drop_last
#     )
#     return dataloader
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
from utils import load_nc_file  # 移除normalize_temperature的导入


class SeaTemperatureDataset(Dataset):
    def __init__(self, data_dir: str, temp_min: float = -5.0, temp_range: float = 45.0):
        """
        海温数据集类 - 支持EN4数据格式
        
        Args:
            data_dir: 包含.nc文件的目录路径
            temp_min: 温度最小值
            temp_range: 温度范围
        """
        self.data_dir = data_dir
        self.temp_min = temp_min
        self.temp_range = temp_range
        
        # 递归查找所有.nc文件
        self.file_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.nc'):
                    self.file_paths.append(os.path.join(root, file))
        
        self.file_paths.sort()
        print(f"Found {len(self.file_paths)} .nc files in {data_dir}")
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .nc files found in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        返回原始海温数据和文件路径
        
        Returns:
            data: 形状为[42, 173, 360]的原始海温张量（摄氏度）
            filepath: 源文件路径
        """
        filepath = self.file_paths[idx]
        
        try:
            # 加载数据
            data = load_nc_file(filepath)
            
            # 确保数据形状正确
            expected_shapes = [(42, 173, 360), (42, 173, 180)]
            
            if data.shape not in expected_shapes:
                # 如果形状不对，尝试调整
                if len(data.shape) == 4 and data.shape[0] == 1:
                    data = data[0]  # 移除时间维度
                    # print(f"Adjusted data shape from {data.shape} to {data.shape}")
                elif len(data.shape) == 3 and data.shape[0] != 42:
                    # 可能需要转置
                    data = np.transpose(data, (2, 0, 1))  # 假设深度在最后一维
                else:
                    print(f"Warning: Unexpected data shape {data.shape} in {filepath}")
            
            # 确保最终形状是[42, 173, 360]
            if data.shape != (42, 173, 360):
                print(f"Error: Cannot process data shape {data.shape} from {filepath}")
                exit(1)
            # print(data)
            if np.nanmean(data) > 50:
                print(f" converting from Kelvin to Celsius")
                data = data - 273.15
            # 转换为tensor
            data = torch.from_numpy(data).float()
            
            # 处理缺失值（NaN）- 使用温度最小值填充
            data = torch.nan_to_num(data, nan=self.temp_min)
            
            # 归一化到[0, 1] - processor期望的输入范围
            data = (data - self.temp_min) / self.temp_range
            data = torch.clamp(data, 0.0, 1.0)
            
            return data, filepath
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # 返回一个零张量作为fallback
            return torch.zeros(42, 173, 360), filepath


def create_dataloader(data_dir: str, batch_size: int, num_workers: int = 4,
                      shuffle: bool = True, **kwargs):
    """
    创建数据加载器
    """
    dataset = SeaTemperatureDataset(data_dir, **kwargs)
    
    # 如果数据集为空，返回None
    if len(dataset) == 0:
        return None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # 测试时不需要drop_last
    )
    return dataloader