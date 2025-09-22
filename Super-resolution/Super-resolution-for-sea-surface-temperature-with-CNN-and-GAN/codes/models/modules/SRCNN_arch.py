import functools
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from .module_util import FiLMLayer

class SRCNN(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_nc,     64, 9, 1, 4, bias=True)
        self.conv2 = nn.Conv2d(   64,     32, 1, 1, 0, bias=True)
        self.conv3 = nn.Conv2d(   32, out_nc, 5, 1, 2, bias=True)

        # activation function
        self.relu = nn.ReLU(inplace=True)
        # === 2. 新增代码: 定义FiLM层 ===
        # 第一个FiLM层，作用于conv1输出的64通道特征
        self.film1 = FiLMLayer(condition_dim=1, feature_channels=64)
        # 第二个FiLM层，作用于conv2输出的32通道特征
        self.film2 = FiLMLayer(condition_dim=1, feature_channels=32)
        # === 代码结束 ===
        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    # def forward(self, x):
    #     h = F.interpolate(x, scale_factor=2, mode='nearest')
    #     h = F.interpolate(h, scale_factor=2, mode='nearest')

    #     h = self.relu(self.conv1(h))
    #     h = self.relu(self.conv2(h))
    #     out = self.relu(self.conv3(h))

    #     return out
    # === 3. 修改: forward方法接收nino参数并应用FiLM ===
    def forward(self, x, nino=None):
        # 保持原始的上采样逻辑
        h = F.interpolate(x, scale_factor=2, mode='nearest')
        h = F.interpolate(h, scale_factor=2, mode='nearest')

        # 第一层卷积和激活
        h = self.relu(self.conv1(h))
        # 应用第一个FiLM层
        if nino is not None:
            h = self.film1(h, nino)

        # 第二层卷积和激活
        h = self.relu(self.conv2(h))
        # 应用第二个FiLM层
        if nino is not None:
            h = self.film2(h, nino)
        
        # 第三层卷积和激活
        out = self.relu(self.conv3(h))

        return out
