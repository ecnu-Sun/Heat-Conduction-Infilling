from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import time
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
class RingBasedPE(nn.Module):
    """
    Ring-based Positional Encoding module for multi-layer ocean temperature fields.
    This module generates a non-parametric positional encoding that combines
    layer (depth) and spatial position information using alternating sin/cos patterns.
    """
    
    def __init__(self, channels=42, height=173, width=360, W_ring=1, T=10000.0):
        """
        Initialize the Ring-based Positional Encoding module.
        
        Args:
            channels: Number of channels (layers/depths)
            height: Height of the spatial grid
            width: Width of the spatial grid
            W_ring: Width of each concentric ring band
            T: Temperature parameter (similar to Transformer PE)
        """
        super().__init__()
        
        # Store dimensions
        self.channels = channels
        self.height = height
        self.width = width
        self.W_ring = W_ring
        self.T = T
        
        # Calculate center coordinates
        center_h = (height - 1) / 2.0
        center_w = (width - 1) / 2.0
        
        # Calculate d_model (max distance from center to corners)
        d_model = math.sqrt(center_h**2 + center_w**2)
        
        # Create coordinate grids
        h_coords = th.arange(height, dtype=th.float32)
        w_coords = th.arange(width, dtype=th.float32)
        h_grid, w_grid = th.meshgrid(h_coords, w_coords, indexing='ij')
        
        # Calculate spatial distances from center for each point
        distances = th.sqrt((h_grid - center_h)**2 + (w_grid - center_w)**2)
        
        # Calculate ring indices
        k_ring = th.floor(distances / W_ring).long()
        
        # Create the positional encoding tensor
        pe = th.zeros(1, channels, height, width, dtype=th.float32)
        
        # For each channel (layer/depth)
        for c in range(channels):
            pos = c  # Layer position
            
            # Calculate theta for all spatial positions
            # θ(pos,i) = pos / T^(i/d_model)
            theta = pos / th.pow(T, distances / d_model)
            
            # Apply alternating sin/cos based on ring index
            # Even rings use sin, odd rings use cos
            pe_channel = th.where(
                k_ring % 2 == 0,
                th.sin(theta),
                th.cos(theta)
            )
            
            pe[0, c, :, :] = pe_channel
        
        # Register as buffer (non-trainable, but moves with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Input tensor with positional encoding added
        """
        scaled_pe = self.pe / 20.0
        return x + scaled_pe.type(x.dtype)
        # return x + self.pe.type(x.dtype)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
    
    ##以下版本是给realtime的嵌入添加门控
    # # 修改：支持传递两个嵌入（扩散时间步嵌入和日期嵌入）
    # def forward(self, x, emb, date_emb=None):
    #     for layer in self:
    #         if isinstance(layer, ResBlock):
    #             # ResBlock 需要两个嵌入
    #             x = layer(x, emb, date_emb)
    #         elif isinstance(layer, TimestepBlock):
    #             x = layer(x, emb)
    #         else:
    #             x = layer(x)
    #     return x
    ##以上版本是给realtime的嵌入添加门控


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1) 

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """
    last_print_time = 0 #用来给gate_weight的打印计时

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels 
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        ##以下用来给realtime的嵌入添加门控，需要分别定义t和data的变为scale和shift的层
        # MLP_t: 处理扩散时间步嵌入
        # self.emb_layers_t = nn.Sequential(
        #     SiLU(),
        #     linear(
        #         emb_channels,
        #         2 * self.out_channels if use_scale_shift_norm else self.out_channels,
        #     ),
        # )
        
        # # MLP_date: 处理日期嵌入
        # self.emb_layers_date = nn.Sequential(
        #     SiLU(),
        #     linear(
        #         emb_channels,
        #         2 * self.out_channels if use_scale_shift_norm else self.out_channels,
        #     ),
        # )
        
        # gate_dim = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        # self.gate_param = nn.Parameter(th.zeros(gate_dim))
        ##以上用来给realtime的嵌入添加门控，需要分别定义t和data的变为scale和shift的层
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)  # channel -> out_channel
        emb_out = self.emb_layers(emb).type(h.dtype)  # emb_channel -> (2)out_channel
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            # 2*out_channel -> out_channel
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            # print("b",end="",flush=True)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            # print("a",end="",flush=True)
            h = self.out_layers(h) # out_channel -> out_channel
        return self.skip_connection(x) + h 
    ##以下版本是给realtime的嵌入添加门控 
    # def forward(self, x, emb, date_emb=None):
    #     """
    #     Apply the block to a Tensor, conditioned on timestep and date embeddings.

    #     :param x: an [N x C x ...] Tensor of features.
    #     :param emb: an [N x emb_channels] Tensor of timestep embeddings.
    #     :param date_emb: an [N x emb_channels] Tensor of date embeddings (optional).
    #     :return: an [N x C x ...] Tensor of outputs.
    #     """
    #     return checkpoint(
    #         self._forward, (x, emb, date_emb), self.parameters(), self.use_checkpoint
    #     )

    # def _forward(self, x, emb, date_emb):
    #     h = self.in_layers(x)  # channel -> out_channel
        
    #     # 修改forward传播逻辑
    #     # 处理扩散时间步嵌入
    #     emb_out_t = self.emb_layers_t(emb).type(h.dtype)
    #     while len(emb_out_t.shape) < len(h.shape):
    #         emb_out_t = emb_out_t[..., None]
        
    #     emb_out_date = self.emb_layers_date(date_emb).type(h.dtype)
    #     while len(emb_out_date.shape) < len(h.shape):
    #         emb_out_date = emb_out_date[..., None]

    #     current_time = time.time()
    #     # if current_time - ResBlock.last_print_time > 60:
    #     #     log_file = "gate_log.txt"
    #     #     with open(log_file, "a") as f:
    #     #         timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
    #     #         f.write(f"--- Log at {timestamp_str} ---\n")
    #     #         f.write(str(self.gate_param.data.detach().cpu()))
    #     #         f.write("\n\n")
    #     #     ResBlock.last_print_time = current_time
    #     #     print(f"use gate weight: {self.gate_param}")
    #     # 计算门控权重 (sigmoid激活，确保在[0,1]范围)
    #     # gate = th.sigmoid(self.gate_param).view(1, -1, *([1] * (len(h.shape) - 2)))
    #     gate = th.sigmoid(self.gate_param).type(h.dtype).view(1, -1, *([1] * (len(h.shape) - 2)))
    #     # 门控融合：根据gate权重混合两个嵌入的影响
    #     emb_out = emb_out_t + gate * emb_out_date
    #     if self.use_scale_shift_norm:
    #         out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
    #         scale, shift = th.chunk(emb_out, 2, dim=1)
    #         h = out_norm(h) * (1 + scale) + shift
    #         h = out_rest(h)
    #     else:
    #         h = h + emb_out
    #         h = self.out_layers(h)
    #     return self.skip_connection(x) + h
    ##以上版本是给realtime的嵌入添加门控

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads  # default 4
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)  
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape  # batch, channel, spatial
        x = x.reshape(b, c, -1) 
        qkv = self.qkv(self.norm(x))
        # qkv: batch, 3*channel, spatial
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2]) # batch*num_heads, 3*channel/num_heads, spatial
        # print('check shape in line 193: ', qkv.shape) 
        h = self.attention(qkv)  # batch*num_heads, channel/num_heads, spatial
        h = h.reshape(b, -1, h.shape[-1]) # batch, channel, spatial
        h = self.proj_out(h) 
        return (x + h).reshape(b, c, *spatial)  


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3 
        q, k, v = th.split(qkv, ch, dim=1) 
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        # batch, T, T
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)  # batch, T, T
        return th.einsum("bts,bcs->bct", weight, v) # batch, C, T

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels  
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        # 时间embed
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        # ==================== Ring-based Positional Encoding 新增代码开始 ====================
        #为多层海洋温度场创建环形位置编码
        # self.use_ring_pe = (in_channels == 42)  # 只在输入为42层时使用
        # if self.use_ring_pe:
        #     self.ring_pe = RingBasedPE(
        #         channels=42,
        #         height=173,
        #         width=360,
        #         W_ring=1,  # 可调整的环带宽度
        #         T=10000.0     # 可调整的温度参数
        #     )
        # ==================== Ring-based Positional Encoding 新增代码结束 ====================
        # ==================== 真实时间嵌入新增代码开始 ====================
        # # 为年份创建嵌入层
        # self.year_embed = nn.Sequential(
        #     linear(model_channels, time_embed_dim),
        #     SiLU(),
        #     linear(time_embed_dim, time_embed_dim),
        # )

        # # 为月份创建嵌入层
        # # 月份是周期性特征 (1月和12月是相邻的)
        # # 我们先将其编码到二维圆上 (sin/cos), 再通过MLP
        # self.month_embed = nn.Sequential(
        #     linear(2, model_channels), # 从(sin, cos) 2维 -> model_channels维
        #     SiLU(),
        #     linear(model_channels, time_embed_dim),
        # )
        # ==================== 真实时间嵌入新增代码结束 ====================   
        # ==================== NINO指数嵌入新增代码开始 ====================
        # # 为NINO指数创建嵌入层
        # self.nino_embed = nn.Sequential(
        #     linear(model_channels, time_embed_dim),
        #     SiLU(),
        #     linear(time_embed_dim, time_embed_dim),
        # )
        # ==================== NINO指数嵌入新增代码结束 ====================     
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # dim卷积的维度，中间channel
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype
    
    def forward(self, x, timesteps, y=None,**kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs. 
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional. 
        :return: an [N x C x ...] Tensor of outputs. 
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        padding_info = []  # Store padding applied at each downsampling step
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # ==================== 真实时间嵌入新增代码开始 ====================
        # year = kwargs.get("year")
        # month = kwargs.get("month")
        # real_time_emb = 0
        # # print(year)
        # # print(month)
        # # 使用与t相同的方式对年份进行编码
        # # 我们将年份减去一个基准值，使其从一个较小的数开始，以获得更好的数值稳定性
        # if year is not None:
        #     base_year = 1849
        #     year_scaled = year - base_year
        #     year_emb_sincos = timestep_embedding(year_scaled, self.model_channels)
        #     real_time_emb = real_time_emb + self.year_embed(year_emb_sincos)
        #     # print(f"use year_scaled:{year_scaled}")
        #     # 将月份(1-12)转换为周期性特征
        #     month_angle = (month.float() - 1) * (2.0 * math.pi / 12.0)
        #     month_sincos = th.cat([th.sin(month_angle)[:,None], th.cos(month_angle)[:,None]], dim=1)
        #     real_time_emb = real_time_emb + self.month_embed(month_sincos)
        #     # print(f"use month_angle:{month_angle}")
        # emb = emb + real_time_emb
        # ==================== 真实时间嵌 入新增代码结束 ====================
        # ==================== NINO指数嵌入新增代码开始 ====================
        # nino = kwargs.get("nino")
        # # print(f"use nino:{nino}")
        # # 将NINO指数乘以500后进行嵌入
        # nino_scaled = nino * 500
        # nino_emb_sincos = timestep_embedding(nino_scaled, self.model_channels)
        # emb = emb + self.nino_embed(nino_emb_sincos)
        # # ==================== NINO指数嵌入新增代码结束 ====================
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        # ==================== 应用Ring-based Positional Encoding 开始 ====================
        # 在输入进入第一个卷积层之前应用位置编码
        # if self.use_ring_pe and h.shape[1] == 42:
        #     if h.shape[-2] == 173 and h.shape[-1] == 360:
        #         # print("usepe")
        #         h = self.ring_pe(h)
        # ==================== 应用Ring-based Positional Encoding 结束 ====================
        for module in self.input_blocks:
            H, W = h.shape[-2:]  # Get current spatial dimensions
            pad_h = (H % 2)      # If odd, pad by 1
            pad_w = (W % 2)
            h = F.pad(h, (0, pad_w, 0, pad_h))  # Pad on right and bottom to make even
            padding_info.append((pad_h, pad_w))  # Store for later use

            h = module(h, emb)
            # print('check shape in line 486: ', h.shape)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            # print('check shape in line 492: ', h.shape)
            cat_in = th.cat([h, hs.pop()], dim=1) 
            h = module(cat_in, emb)  

            pad_h, pad_w = padding_info.pop()  # Retrieve last stored padding values
            if pad_h > 0 or pad_w > 0:
                h = h[..., : -pad_h if pad_h > 0 else None, : -pad_w if pad_w > 0 else None]
        h = h.type(x.dtype)
        return self.out(h)
    
    ##以下这个forword方法是用于给realtime的嵌入添加门控
    # def forward(self, x, timesteps, y=None, **kwargs):
    #     """
    #     Apply the model to an input batch.

    #     :param x: an [N x C x ...] Tensor of inputs. 
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param y: an [N] Tensor of labels, if class-conditional. 
    #     :return: an [N x C x ...] Tensor of outputs. 
    #     """
    #     assert (y is not None) == (
    #         self.num_classes is not None
    #     ), "must specify y if and only if the model is class-conditional"

    #     hs = []
    #     padding_info = []  # Store padding applied at each downsampling step
        
    #     # 扩散时间步嵌入
    #     emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
    #     # 真实时间嵌入（年份和月份）- 现在分开处理
    #     date_emb = None
    #     year = kwargs.get("year")
    #     month = kwargs.get("month")
        
    #     if year is not None or month is not None:
    #         date_emb = 0
    #         if year is not None:
    #             # 使用与t相同的方式对年份进行编码
    #             base_year = 1849 
    #             year_scaled = year - base_year
    #             year_emb_sincos = timestep_embedding(year_scaled, self.model_channels)
    #             date_emb = date_emb + self.year_embed(year_emb_sincos)
                
    #         if month is not None:
    #             # 将月份(1-12)转换为周期性特征
    #             month_angle = (month.float() - 1) * (2.0 * math.pi / 12.0)
    #             month_sincos = th.cat([th.sin(month_angle)[:,None], th.cos(month_angle)[:,None]], dim=1)
    #             date_emb = date_emb + self.month_embed(month_sincos)
        
    #     if self.num_classes is not None:
    #         assert y.shape == (x.shape[0],)
    #         emb = emb + self.label_emb(y)

    #     h = x.type(self.inner_dtype)
    #     for module in self.input_blocks:
    #         H, W = h.shape[-2:]  # Get current spatial dimensions
    #         pad_h = (H % 2)      # If odd, pad by 1
    #         pad_w = (W % 2)
    #         h = F.pad(h, (0, pad_w, 0, pad_h))  # Pad on right and bottom to make even
    #         padding_info.append((pad_h, pad_w))  # Store for later use

    #         # 传递两个嵌入
    #         h = module(h, emb, date_emb)
    #         hs.append(h)
            
    #     h = self.middle_block(h, emb, date_emb)
        
    #     for module in self.output_blocks:
    #         cat_in = th.cat([h, hs.pop()], dim=1) 
    #         h = module(cat_in, emb, date_emb)  

    #         pad_h, pad_w = padding_info.pop()  # Retrieve last stored padding values
    #         if pad_h > 0 or pad_w > 0:
    #             h = h[..., : -pad_h if pad_h > 0 else None, : -pad_w if pad_w > 0 else None]
                
    #     h = h.type(x.dtype)
    #     return self.out(h)
    ##以上这个forword方法是用于给realtime的嵌入添加门控

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)  
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)

