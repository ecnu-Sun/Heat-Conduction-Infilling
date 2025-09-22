"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import time

import numpy as np
import torch as th
import blobfile as bf
import torch.distributed as dist
import torch.nn.functional as F

import xarray as xr

from improved_diffusion import dist_util, logger
from improved_diffusion.gaussian_diffusion import _extract_into_tensor
from improved_diffusion.script_util_v2 import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    # classifier_defaults,  # Added
    create_model_and_diffusion,
    # create_classifier, # Added
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.guided_util import *


def get_gaussian_kernel(size, sigma):
    coords = th.arange(size, dtype=th.float32)
    coords -= size // 2
    g = th.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g.outer(g)
    # return (g / g.sum()).view(1, 1, size, size)  
    return g.view(1, 1, size, size)

# 要替换成的代码
def get_gaussian_kernel_3d(size, sigma):
    """生成一个3D高斯核用于3D卷积"""
    # 首先创建1D高斯分布
    coords = th.arange(size, dtype=th.float32)
    coords -= size // 2
    g = th.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum() # 归一化，确保模糊后能量守恒

    # 利用广播机制从1D向量生成3D核
    g_d = g.view(size, 1, 1)
    g_h = g.view(1, size, 1)
    g_w = g.view(1, 1, size)
    kernel = g_d * g_h * g_w
    kernel = kernel / kernel.sum()

    # 塑造成 (out_channels, in_channels, D, H, W) 的5D形状
    return kernel.view(1, 1, size, size, size)

def depth_adaptive_blur(gradient, depth_levels, kernel_size=5, sigma=1.0):
    """
    对梯度进行深度自适应的高斯模糊
    
    Args:
        gradient: 输入梯度张量，形状为 (B, D, H, W)
        depth_levels: 海洋深度数组
        kernel_size: 水平方向的核大小
        sigma: 高斯核的标准差
    """
    B, D, H, W = gradient.shape
    device = gradient.device
    
    # 水平方向使用标准高斯核
    coords = th.arange(kernel_size, dtype=th.float32, device=device)
    coords -= kernel_size // 2
    g_spatial = th.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_spatial = g_spatial / g_spatial.sum()
    
    # 创建2D高斯核用于水平模糊
    kernel_2d = th.outer(g_spatial, g_spatial)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    # 首先对每个深度层进行水平模糊
    gradient_reshaped = gradient.view(B * D, 1, H, W)
    gradient_h_blurred = F.conv2d(
        gradient_reshaped,
        kernel_2d,
        padding=kernel_size // 2
    )
    gradient_h_blurred = gradient_h_blurred.view(B, D, H, W)
    
    # 然后进行深度方向的自适应模糊
    output = th.zeros_like(gradient_h_blurred)

    
    # 对每个深度层计算加权平均
    for d in range(D):
        # 计算当前层与邻近层的高斯权重
        weights = th.zeros(D, device=device)
        
        for d_neighbor in range(max(0, d-2), min(D, d+3)):  # 考虑上下各2层
            if d_neighbor == d:
                weights[d] = 1.0  # 自身权重最高
            else:
                # 计算物理距离
                distance = abs(depth_levels[d] - depth_levels[d_neighbor])
                # 使用物理距离计算权重，这里50是一个缩放因子，可以调整
                weights[d_neighbor] = th.exp(-(distance ** 2) / 1000000)
        
        # 归一化权重
        weights = weights / weights.sum()
        
        # 应用权重
        for d_neighbor in range(D):
            if weights[d_neighbor] > 1e-6:  # 只处理有意义的权重
                output[:, d] += weights[d_neighbor] * gradient_h_blurred[:, d_neighbor]
    
    return output


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())  # Unet Model(用于去噪) 和 Gaussian Diffusion
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("Loading sparse data for guiding...")
    guided_arr_dict = {}
    for file in bf.listdir(args.sparse_data_path):
        # file only contains the name of the file, not the full path
        # print(file[-11:-9])
        # if file.endswith(".nc"):
        #     # print(file)
        #     path = os.path.join(args.sparse_data_path, file)
        #     ds = xr.open_dataset(path)
        #     arr = ds['temperature'].values - 273.15
        #     arr = arr.astype(np.float32)
        #     if len(arr.shape) == 4:
        #         arr = arr.reshape(arr.shape[1], arr.shape[2], arr.shape[3])

        #     guided_arr_dict[str(file)[0:-3]] = arr  # .npy->.nc
        # 跳过非 .nc 文件
        if not file.endswith(".nc"):
            continue

        print(f"--> Loading guidance file: {file}")
        path = os.path.join(args.sparse_data_path, file)
        
        try:
            with xr.open_dataset(path) as ds:
                # 1. 变量选择：优先'thetao'，其次'temperature'
                var_name = None
                if 'thetao' in ds.data_vars:
                    var_name = 'thetao'
                elif 'temperature' in ds.data_vars:
                    var_name = 'temperature'
                else:
                    print(f"    错误：在文件 {file} 中找不到 'thetao' 或 'temperature' 变量。跳过此文件。")
                    continue
                
                print(f"    使用变量: '{var_name}'")
                arr = ds[var_name].values.astype(np.float32)
                arr = arr.squeeze()
                # 2. 单位转换：根据数值大小判断
                mean_val = np.nanmean(arr)
                if mean_val > 100:
                    # print(f"    检测到数据平均值 ({mean_val:.2f}) > 100，自动进行开尔文到摄氏度的单位转换。")
                    arr = arr - 273.15
                else:
                    # print(f"    数据平均值 ({mean_val:.2f}) 处于正常摄氏度范围，不进行单位转换。")
                    pass

                # 将处理好的数据存入字典
                guided_arr_dict[file[:-3]] = arr

        except Exception as e:
            print(f"    处理文件 {file} 时发生错误: {e}")
    print(guided_arr_dict.keys())
    ### landmark
    first_data_key = next(iter(guided_arr_dict))
    land_mask_np = np.isnan(guided_arr_dict[first_data_key])
    land_mask = th.from_numpy(land_mask_np).to(dist_util.dev())
    ### landmark
    logger.log("Successfully load the guided data!")
    depth_levels = th.tensor([5.021590e+00, 1.507854e+01, 2.516046e+01, 3.527829e+01, 4.544776e+01,
                         5.569149e+01, 6.604198e+01, 7.654591e+01, 8.727029e+01, 9.831118e+01,
                         1.098062e+02, 1.219519e+02, 1.350285e+02, 1.494337e+02, 1.657285e+02,
                         1.846975e+02, 2.074254e+02, 2.353862e+02, 2.705341e+02, 3.153741e+02,
                         3.729655e+02, 4.468009e+02, 5.405022e+02, 6.573229e+02, 7.995496e+02,
                         9.679958e+02, 1.161806e+03, 1.378661e+03, 1.615291e+03, 1.868071e+03,
                         2.133517e+03, 2.408583e+03, 2.690780e+03, 2.978166e+03, 3.269278e+03,
                         3.563041e+03, 3.858676e+03, 4.155628e+03, 4.453502e+03, 4.752021e+03,
                         5.050990e+03, 5.350272e+03], dtype=th.float32).cuda()
    softmask_kernel = get_gaussian_kernel(5, 1.0).float().to("cuda")

    # print(softmask_kernel)

    def cond_fn(x, t, p_mean_var, y=None,**kwargs):
        """
        这是生成条件引导梯度矩阵的函数，p_mean_var['pred_xstart']是x0hat，y是ground truth，可选高斯核模糊或者否
        """
        assert y is not None
        # print(f"Shape of y: {y.shape}")
        # print(f"Shape of pred_xstart: {x.shape}")
        x = p_mean_var['pred_xstart']  # x0hat
        s = args.grad_scale
        gradient = 2 * (y - x)
        gradient = th.nan_to_num(gradient, nan=0.0)
        if args.use_softmask:
            # 原始2d卷积代码：
            size = softmask_kernel.shape[-1]
            gradient = F.conv2d(
                gradient,
                softmask_kernel.expand(gradient.shape[1], 1, size, size),
                stride=1,
                padding=size // 2,
                groups=gradient.shape[1]
            )

        # copy_weight = 0.9
        # original_gradient_after_2d_conv = gradient.clone()
        # _B, D, _H, _W = gradient.shape
        # for d in range(D):
        #     source_mask = original_gradient_after_2d_conv[:, d, :, :] != 0
        #     if not source_mask.any():
        #         continue
        #     grad_to_copy = original_gradient_after_2d_conv[:, d, :, :] * copy_weight
        #     # --- 向上复制 ---
        #     if d > 0:
        #         target_mask_up = gradient[:, d - 1, :, :] == 0
        #         final_mask_up = source_mask & target_mask_up
        #         gradient[:, d - 1, :, :][final_mask_up] = grad_to_copy[final_mask_up]
        #     # --- 向下复制 ---
        #     if d < D - 1:
        #         target_mask_down = gradient[:, d + 1, :, :] == 0
        #         final_mask_down = source_mask & target_mask_down
        #         gradient[:, d + 1, :, :][final_mask_down] = grad_to_copy[final_mask_down]
                
            # size = softmask_kernel.shape[-1]
            # grad_5d = gradient.unsqueeze(1)
            # # 使用3D卷积进行模糊
            # blurred_grad_5d = th.nn.functional.conv3d(
            #     grad_5d,
            #     softmask_kernel,  # 3D核的形状应为 (1, 1, size, size, size)
            #     stride=1,
            #     padding=size // 2,
            #     groups=1
            # )
            # # 将形状还原为 (B, D, H, W)
            # gradient = blurred_grad_5d.squeeze(1)
            
            #深度适应3d卷积：
            # gradient = depth_adaptive_blur(gradient, depth_levels, kernel_size=5, sigma=1.0)
        return gradient * s

    def create_pigdm_cond_fn(model, diffusion, args):
        """
        创建ΠGDM条件函数
        """
        def cond_fn(x, t, p_mean_var, y=None):
            assert y is not None
            
            # 重新计算x_hat以建立计算图
            with th.enable_grad():
                x_in = x.detach().requires_grad_()
                
                # 通过模型计算epsilon
                model_output = model(x_in, diffusion._scale_timesteps(t))
                print((t* diffusion.num_timesteps / 1000.0).long())
                # 从epsilon计算x0
                x_hat = diffusion._predict_xstart_from_eps(x_t=x_in, t=(t* diffusion.num_timesteps / 1000.0).long(), eps=model_output)
                # print(x_hat)
                # exit(0)
                # 创建mask
                mask = ~th.isnan(y)
                # 计算 H†y - H†H(x̂t)
                diff = th.zeros_like(x_hat)
                diff[mask] = y[mask] - x_hat[mask]
                
                # 计算雅可比向量积
                scalar_product = (diff.detach() * x_hat).sum() 
                print(diff.sum())
                # print(f"Scalar product: {scalar_product.item()}")
                gradient = th.autograd.grad(scalar_product, x_in, retain_graph=True)[0]
                
                # 处理NaN
                gradient = th.nan_to_num(gradient, nan=0.0)
                # if args.use_softmask:
                #     size = softmask_kernel.shape[-1]
                #     gradient = F.conv2d(
                #         gradient,
                #         softmask_kernel.expand(gradient.shape[1], 1, size, size),
                #         stride=1,
                #         padding=size // 2,
                #         groups=gradient.shape[1]
                #     )
                # VP-SDE缩放
                # alpha_bar_t = _extract_into_tensor(
                #     diffusion.alphas_cumprod, (t* diffusion.num_timesteps / 1000.0).long(), x.shape
                # )
                # gradient = gradient * th.sqrt(alpha_bar_t)
                alphas_cumprod_sqrt = th.sqrt(_extract_into_tensor(diffusion.alphas_cumprod,(t* diffusion.num_timesteps / 1000.0).long(), x.shape) )  
                print(f"Gradient Abs Sum: {th.abs(gradient * args.grad_scale * alphas_cumprod_sqrt).sum().item()}")
            return gradient * args.grad_scale * alphas_cumprod_sqrt
    
        return cond_fn

    def cond_fn_3d(x, t, p_mean_var, y=None):
        assert y is not None
        x = p_mean_var['pred_xstart']  # x0
        s = args.grad_scale
        gradient = 2 * (y - x)
        gradient = th.nan_to_num(gradient, nan=0.0)

        if args.use_softmask:
            size = softmask_kernel.shape[-1]
            gradient = gradient.unsqueeze(1)
            gradient = F.conv3d(
                gradient,
                softmask_kernel,
                stride=1,
                padding=size // 2,
                groups=1
            )  # B1CHW 
            gradient = gradient.squeeze(1)
        return gradient * s

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None,**kwargs)
    # def model_fn(x, t, **kwargs):
    #     assert "y" in kwargs
    #     return model(x, t, **kwargs)

    # outputdir
    date = time.strftime("%m%d")
    if args.dynamic_guided:
        config = f"dyn_next={args.dynamic_guided_with_next}_r={args.guided_rate}_sigma={args.use_sigma}"
    else:
        config = f"s={args.grad_scale}_r={args.guided_rate}_loss={args.loss_model}_softmask={args.use_softmask}_sigma={args.use_sigma}"

    out_dir = os.path.dirname(os.environ.get("DIFFUSION_SAMPLE_LOGDIR", logger.get_dir()))
    out_dir = os.path.join(out_dir, date, config)
    os.makedirs(out_dir, exist_ok=True)

    logger.log("sampling...")

    loss_preds = []
    loss_guideds = []
    losses = []
    for key, guided_arr in guided_arr_dict.items():
        all_images = []
        all_loss_pred = []
        all_loss_guided = []
        all_loss = []
        logger.log(f"sampling {key}...")
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            guided_y, eval_y = split_guided_eval_batch_size(args.batch_size, guided_arr, args.guided_rate)
            model_kwargs["y"] = normalization(guided_y)
            
            ###传入landmask
            model_kwargs["land_mask"] = land_mask.unsqueeze(0).expand_as(model_kwargs["y"])
            ###传入landmask

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            # 传递realtime开始
            date_str = key[-6:]
            year = int(date_str[:4])
            month = int(date_str[4:])
            year_tensor = th.tensor([year] * args.batch_size, device=dist_util.dev())
            month_tensor = th.tensor([month] * args.batch_size, device=dist_util.dev())
            model_kwargs["year"] = year_tensor
            model_kwargs["month"] = month_tensor
            # 传递realtime结束
            
            # sample = sample_fn(
            #     model_fn,  # 由于去噪的U-net网络
            #     (args.batch_size, args.in_channels, args.image_size_H, args.image_size_W),  # 形状，第二个已修改
            #     clip_denoised=args.clip_denoised,
            #     # 预测出的 x̂₀ 的数值范围可能会略微超出[-1, 1],设置这个参数为true可以让任何大于1的值都变成1，任何小于-1的值都变成-1
            #     model_kwargs=model_kwargs,
            #     cond_fn=create_pigdm_cond_fn(model, diffusion, args),  # 生成条件引导梯度矩阵的函数
            #     use_sigma=args.use_sigma,
            #     dynamic_guided=args.dynamic_guided,  # 这个参数文章里的方法没用到，是另一个引导方法
            #     dynamic_guided_with_next=args.dynamic_guided_with_next,  # 这个参数文章里的方法没用到，是另一个引导方法
            #     device=dist_util.dev(),
            # )
            sample_kwargs = {
                "clip_denoised": args.clip_denoised,
                "model_kwargs": model_kwargs,
                # "cond_fn": create_pigdm_cond_fn(model, diffusion, args),
                "cond_fn":cond_fn,
                "device": dist_util.dev(),
            }

            # 2. 如果不使用 DDIM (即使用DDPM)，才添加那几个额外参数
            if not args.use_ddim:
                sample_kwargs["use_sigma"] = args.use_sigma
                sample_kwargs["dynamic_guided"] = args.dynamic_guided
                sample_kwargs["dynamic_guided_with_next"] = args.dynamic_guided_with_next
            
            # print(f"Sample kwargs: {sample_kwargs}")
            # 3. 使用字典解包的方式调用函数
            sample = sample_fn(
                model_fn,
                (args.batch_size, args.in_channels, args.image_size_H, args.image_size_W),
                **sample_kwargs,
            )

            sample = ((sample + 1) * 22.5 - 5).clamp(-5, 40)  # 把去噪模型输出的[-1, 1]反归一化到-5-40度
            sample = sample.permute(0, 2, 3, 1)  # 将维度重新排列为 NHWC 格式
            sample = sample.contiguous()  # 像 permute 这样的操作虽然改变了张量的维度信息，但并不会真的在内存中移动数据，
            # 这可能导致张量在内存中的存储变得不连续，因此此处创建一个与原张量数据相同，但保证在内存中是连续存储的新张量

            gathered_samples = [th.zeros_like(sample) for _ in
                                range(dist.get_world_size())]  # 创建一个列表 gathered_samples，其中包含和任务总进程数（GPU数）一样多个的占位符
            dist.all_gather(gathered_samples,
                            sample)  # gather not supported with NCCL，把自己计算出的 sample 发送给所有其他的GPU。同时，收来自所有其他GPU的 sample 张量

            loss_pred = calculate_loss(sample, eval_y, args.loss_model)
            loss_guided = calculate_loss(sample, guided_y, args.loss_model)
            loss_total = (1 - args.guided_rate) * loss_pred + args.guided_rate * loss_guided  # 根据引导点和评估点在整个数据场中的面积占比，对前面计算出的两个损失进行加权平均
            all_loss_pred.append(loss_pred)
            all_loss_guided.append(loss_guided)
            all_loss.append(loss_total)

            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

            logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        pred_loss = np.mean(all_loss_pred, axis=0)
        guided_loss = np.mean(all_loss_guided, axis=0)
        total_loss = np.mean(all_loss, axis=0)
        loss_preds.append(pred_loss)
        loss_guideds.append(guided_loss)
        losses.append(total_loss)

        if dist.get_rank() == 0:  # 只让0号进程写日志，注意，这里的报告的pred_loss和guided_loss只是0号进程处理的samples的，不过单GPU下没问题
            logger.log(f"Loss of {key} pred_loss: {pred_loss}, guided_loss: {guided_loss}, total_loss: {total_loss}")
            shape_str = "x".join([str(x) for x in arr.shape])
            # outputpath
            out_path = os.path.join(out_dir, f"{key}_sample{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr)
            logger.log(f"sampling {key} complete")

    dist.barrier()  # 所有进程运行到此处后才继续
    logger.log("Complete All Sample!")
    logger.log(
        f"Compute Total Loss: pred_loss: {np.mean(loss_preds, axis=0)}, guided_loss: {np.mean(loss_guideds, axis=0)}, total_loss: {np.mean(losses, axis=0)}")


def create_argparser():
    defaults = dict(
        image_size_H=173,
        image_size_W=360,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        use_sigma=True,
        model_path="",
        sparse_data_path="",
        grad_scale=5.0,  # when 0: sample from the base diffusion model
        use_softmask=True,
        dynamic_guided=False,
        dynamic_guided_with_next=False,
        guided_rate=0.075,
        loss_model="l1",
        use_fp16=True,
    )
    defaults.update(model_and_diffusion_defaults())
    # defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

    if dist.is_initialized():
        dist.destroy_process_group()
