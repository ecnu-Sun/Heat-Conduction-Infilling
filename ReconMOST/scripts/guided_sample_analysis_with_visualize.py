# """
# Like image_sample.py, but use a noisy image classifier to guide the sampling
# process towards more realistic images.
# """

# import argparse
# import os
# import time

# import numpy as np
# import torch as th
# import blobfile as bf
# import torch.distributed as dist
# import torch.nn.functional as F

# import xarray as xr

# import matplotlib
# matplotlib.use("Agg") 
# import matplotlib.pyplot as plt

# from improved_diffusion import dist_util, logger
# from improved_diffusion.gaussian_diffusion import _extract_into_tensor
# from improved_diffusion.script_util_v2 import (
#     NUM_CLASSES,
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict,
# )
# from improved_diffusion.guided_util import *
# th.set_printoptions(profile="full")

# # --- 1. 新增的可视化函数 ---
# def visualize_and_save_frame(tensor_xt, timestep, save_dir):
#     """
#     可视化单个时间步的xt张量（取batch中的第一个样本）并保存为图像。
#     """
#     # 从batch中取出第一个样本的第一层
#     sample = tensor_xt[0] 
#     sample = sample[0]
#     # 反归一化：从[-1, 1]范围转换回温度范围[-5, 40]
#     sample = ((sample + 1) * 22.5 - 5).clamp(-5, 40)
    
#     sample_np = sample.cpu().numpy()

#     # 绘图
#     plt.figure(figsize=(10, 5)) # 设置图像大小
#     im = plt.imshow(sample_np, cmap='coolwarm', vmin=-5, vmax=40, origin='lower')
#     plt.title(f"Timestep t={timestep}")
#     plt.axis('off')

#     # 添加颜色条
#     cbar = plt.colorbar(im, orientation='vertical', pad=0.02)
#     cbar.set_label('Temperature (°C)')
    
#     # 保存图像帧
#     frame_path = os.path.join(save_dir, f"frame_{timestep:04d}.png")
#     plt.savefig(frame_path, bbox_inches='tight', dpi=100) # dpi可以调整清晰度
#     plt.close() # 关闭图像以释放内存
# # ------------------------------


# def get_gaussian_kernel(size, sigma):
#     coords = th.arange(size, dtype=th.float32)
#     coords -= size // 2
#     g = th.exp(-(coords ** 2) / (2 * sigma ** 2))
#     g = g.outer(g)
#     return g.view(1, 1, size, size)


# def main():
#     args = create_argparser().parse_args()

#     dist_util.setup_dist()
#     logger.configure()

#     logger.log("creating model and diffusion...")
#     model, diffusion = create_model_and_diffusion(
#         **args_to_dict(args, model_and_diffusion_defaults().keys())
#     )
#     model.load_state_dict(
#         dist_util.load_state_dict(args.model_path, map_location="cpu")
#     )
#     model.to(dist_util.dev())
#     if args.use_fp16:
#         model.convert_to_fp16()
#     model.eval()

#     logger.log("Loading sparse data for guiding...")
#     guided_arr_dict = {}
#     for file in bf.listdir(args.sparse_data_path):
#         if not file.endswith(".nc"):
#             continue

#         print(f"--> Loading guidance file: {file}")
#         path = os.path.join(args.sparse_data_path, file)
        
#         try:
#             with xr.open_dataset(path) as ds:
#                 var_name = 'thetao' if 'thetao' in ds.data_vars else 'temperature'
#                 if var_name not in ds.data_vars:
#                     print(f"    错误：在文件 {file} 中找不到 'thetao' 或 'temperature' 变量。跳过此文件。")
#                     continue
                
#                 #print(f"    使用变量: '{var_name}'")
#                 arr = ds[var_name].values.astype(np.float32).squeeze()
                
#                 mean_val = np.nanmean(arr)
#                 if mean_val > 100:
#                     arr = arr - 273.15
                
#                 guided_arr_dict[file[:-3]] = arr
#         except Exception as e:
#             print(f"    处理文件 {file} 时发生错误: {e}")

#     print(guided_arr_dict.keys())
#     logger.log("Successfully load the guided data!")

# softmask_kernel = get_gaussian_kernel(5, 1.0).float().to("cuda")

# def cond_fn(x, t, p_mean_var, y=None):
#     assert y is not None
#     x = p_mean_var['pred_xstart']
#     s = args.grad_scale
#     gradient = 2 * (y - x)
#     gradient = th.nan_to_num(gradient, nan=0.0)
#     if args.use_softmask:
#         size = softmask_kernel.shape[-1]
#         gradient = F.conv2d(
#             gradient,
#             softmask_kernel.expand(gradient.shape[1], 1, size, size),
#             stride=1,
#             padding=size // 2,
#             groups=gradient.shape[1]
#         )
#     return gradient * s

#     def create_pigdm_cond_fn(model, diffusion, args):
#         def cond_fn(x, t, p_mean_var, y=None, land_mask=None):
#             assert y is not None
#             assert land_mask is not None, "Land mask must be provided via model_kwargs"
#             with th.enable_grad():
#                 x_in = x.detach().requires_grad_()
#                 model_output = model(x_in, diffusion._scale_timesteps(t))
#                 t_long = (t * diffusion.num_timesteps / 1000.0).long()
#                 x_hat = diffusion._predict_xstart_from_eps(x_t=x_in, t=t_long, eps=model_output)
#                 mask = ~th.isnan(y) #引导点的地方设置为 True
#                 print(mask[0])
#                 diff = th.zeros_like(x_hat)
#                 diff[mask] = y[mask] - x_hat[mask]
#                 print(diff[0])
#                 scalar_product = (diff.detach() * x_hat).sum() 
#                 gradient = th.autograd.grad(scalar_product, x_in, retain_graph=False)[0] # retain_graph=False
#                 print(gradient[0])
#                 print(f"Gradient Abs Sum: {th.abs(gradient).sum().item()}")
#                 gradient[land_mask] = 0.0
#                 print(f"Gradient Abs Sum: {th.abs(gradient).sum().item()}")
#                 gradient = th.nan_to_num(gradient, nan=0.0)
#                 alphas_cumprod_sqrt = th.sqrt(_extract_into_tensor(diffusion.alphas_cumprod, t_long, x.shape))
#                 exit(1)
#             return gradient * args.grad_scale * alphas_cumprod_sqrt
#         return cond_fn

#     def model_fn(x, t, y=None, **kwargs):
#         assert y is not None
#         return model(x, t, y if args.class_cond else None)

#     date = time.strftime("%m%d")
#     config = f"s={args.grad_scale}_r={args.guided_rate}_loss={args.loss_model}_softmask={args.use_softmask}_sigma={args.use_sigma}"
#     out_dir = os.path.join(os.environ.get("DIFFUSION_SAMPLE_LOGDIR", logger.get_dir()), date, config)
#     os.makedirs(out_dir, exist_ok=True)

#     logger.log("sampling...")

#     loss_preds, loss_guideds, losses = [], [], []
#     for key, guided_arr in guided_arr_dict.items():
#         all_images, all_loss_pred, all_loss_guided, all_loss = [], [], [], []
#         logger.log(f"sampling {key}...")
#         land_mask_np = np.isnan(guided_arr)
#         while len(all_images) * args.batch_size < args.num_samples:
#             model_kwargs = {}
#             guided_y, eval_y = split_guided_eval_batch_size(args.batch_size, guided_arr, args.guided_rate)
#             model_kwargs["y"] = normalization(guided_y)
#             # 2. 将numpy掩码转换为Tensor，并扩展到与batch大小一致
#             land_mask_batch = th.from_numpy(
#                 np.stack([land_mask_np] * args.batch_size)
#             ).to(dist_util.dev())
            
#             # 3. 将陆地掩码放入 model_kwargs 中，以便传递给引导函数
#             model_kwargs["land_mask"] = land_mask_batch
#             # --- 2. 替换采样循环 ---
#             # 定义视频帧保存目录
#             video_frames_dir = os.path.join(out_dir, f"{key}_video_frames_{int(time.time())}")
#             os.makedirs(video_frames_dir, exist_ok=True)
#             logger.log(f"Saving visualization frames to {video_frames_dir}")

#             # 初始噪声，即 t=T 时的 x_t
#             initial_noise = th.randn(
#                 (args.batch_size, args.in_channels, args.image_size_H, args.image_size_W),
#                 device=dist_util.dev()
#             )
            
#             # 可视化第一帧（纯噪声）
#             visualize_and_save_frame(initial_noise, diffusion.num_timesteps - 1, video_frames_dir)

#             # 获取渐进式采样器
#             sample_generator = diffusion.p_sample_loop_progressive(
#                 model_fn,
#                 (args.batch_size, args.in_channels, args.image_size_H, args.image_size_W),
#                 noise=initial_noise, # 从相同的初始噪声开始
#                 clip_denoised=args.clip_denoised,
#                 model_kwargs=model_kwargs,
#                 cond_fn= create_pigdm_cond_fn(model, diffusion, args),
#                 use_sigma=args.use_sigma,
#                 device=dist_util.dev(),
#             )

#             final_sample = None
#             indices = list(range(diffusion.num_timesteps))[::-1]

#             # 遍历每一步的采样结果
#             for i, result in enumerate(sample_generator):
#                 timestep = indices[i]
#                 final_sample = result["sample"] # 'sample' 是 x_{t-1}
#                 # 可视化当前步骤的结果 (x_{t-1})
#                 if timestep > 0: # 避免可视化不存在的-1步
#                     visualize_and_save_frame(final_sample, timestep - 1, video_frames_dir)
            
#             # 循环结束后，final_sample 就是最终的生成结果
#             sample = final_sample
#             # --------------------------

#             sample = ((sample + 1) * 22.5 - 5).clamp(-5, 40)
#             sample = sample.permute(0, 2, 3, 1).contiguous()

#             gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
#             dist.all_gather(gathered_samples, sample)

#             loss_pred = calculate_loss(sample, eval_y, args.loss_model)
#             loss_guided = calculate_loss(sample, guided_y, args.loss_model)
#             loss_total = (1 - args.guided_rate) * loss_pred + args.guided_rate * loss_guided
#             all_loss_pred.append(loss_pred)
#             all_loss_guided.append(loss_guided)
#             all_loss.append(loss_total)

#             all_images.extend([s.cpu().numpy() for s in gathered_samples])
#             logger.log(f"created {len(all_images) * args.batch_size} samples")
            
#             # 为了演示，我们只生成一个batch并可视化，所以在此跳出
#             if args.num_samples <= args.batch_size:
#                 break


#         arr = np.concatenate(all_images, axis=0)
#         pred_loss = np.mean(all_loss_pred, axis=0)
#         guided_loss = np.mean(all_loss_guided, axis=0)
#         total_loss = np.mean(all_loss, axis=0)
#         loss_preds.append(pred_loss)
#         loss_guideds.append(guided_loss)
#         losses.append(total_loss)

#         if dist.get_rank() == 0:
#             logger.log(f"Loss of {key} pred_loss: {pred_loss}, guided_loss: {guided_loss}, total_loss: {total_loss}")
#             shape_str = "x".join([str(x) for x in arr.shape])
#             out_path = os.path.join(out_dir, f"{key}_sample{shape_str}.npz")
#             logger.log(f"saving to {out_path}")
#             np.savez(out_path, arr)
#             logger.log(f"sampling {key} complete")
        
#         # 为了演示，我们只处理一个 guided_arr
#         break


#     dist.barrier()
#     logger.log("Complete All Sample!")
#     logger.log(f"Compute Total Loss: pred_loss: {np.mean(loss_preds, axis=0)}, guided_loss: {np.mean(loss_guideds, axis=0)}, total_loss: {np.mean(losses, axis=0)}")


# def create_argparser():
#     defaults = dict(
#         image_size_H=173,
#         ima_denoised=True,
#         image_size_W=360,
#         clipsamples=1, # 为了快速演示，只生成一个样本
#         batch_size=1, # batch size 也设为1
#         num_samples=1,
#         use_ddim=False,
#         use_sigma=True,
#         model_path="",
#         sparse_data_path="",
#         grad_scale=1.0, 
#         use_softmask=False,
#         dynamic_guided=False,
#         dynamic_guided_with_next=False,
#         guided_rate=0.6,
#         loss_model="l1",
#         use_fp16=False,
#     )
#     defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser


# if __name__ == "__main__":
#     main()
#     if dist.is_initialized():
#         dist.destroy_process_group()
import argparse
import os
import time

import numpy as np
import torch as th
import blobfile as bf
import torch.distributed as dist
import torch.nn.functional as F
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from improved_diffusion import dist_util, logger
from improved_diffusion.gaussian_diffusion import _extract_into_tensor
from improved_diffusion.script_util_v2 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.guided_util import *
th.set_printoptions(profile="full")

def visualize_tensor_map(tensor, title, save_path, cmap='coolwarm', vmin=None, vmax=None):
    """可视化张量并保存为地图"""
    if tensor.dim() == 4:
        tensor = tensor[0, 0]  # 取第一个batch和第一个channel
    elif tensor.dim() == 3:
        tensor = tensor[0]  # 取第一个batch
    
    tensor_np = tensor.cpu().detach().numpy()
    
    plt.figure(figsize=(12, 6))
    im = plt.imshow(tensor_np, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    plt.title(title)
    plt.colorbar(im, orientation='vertical', pad=0.02)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
def visualize_and_save_frame(tensor_xt, timestep, save_dir):
    sample = tensor_xt[0, 0]
    sample = ((sample + 1) * 22.5 - 5).clamp(-5, 40)
    sample_np = sample.cpu().numpy()

    plt.figure(figsize=(10, 5))
    im = plt.imshow(sample_np, cmap='coolwarm', vmin=-5, vmax=40, origin='lower')
    plt.title(f"Timestep t={timestep}")
    plt.axis('off')
    
    cbar = plt.colorbar(im, orientation='vertical', pad=0.02)
    cbar.set_label('Temperature (°C)')
    
    frame_path = os.path.join(save_dir, f"frame_{timestep:04d}.png")
    plt.savefig(frame_path, bbox_inches='tight', dpi=100)
    plt.close()


def get_gaussian_kernel(size, sigma):
    coords = th.arange(size, dtype=th.float32)
    coords -= size // 2
    g = th.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g.outer(g)
    return g.view(1, 1, size, size)

def create_boundary_mask(shape, border_width=2):
    """创建边界掩码，边界处为True"""
    batch_size, channels, height, width = shape
    mask = th.zeros((batch_size, channels, height, width), dtype=th.bool)
    
    # 标记边界
    mask[:, :, :border_width, :] = True  # 上边界
    mask[:, :, -border_width:, :] = True  # 下边界
    mask[:, :, :, :border_width] = True  # 左边界
    mask[:, :, :, -border_width:] = True  # 右边界
    
    return mask

# def create_pigdm_cond_fn(model, diffusion, args, vis_dir='./viewdata'):
#     def cond_fn(x, t, p_mean_var, y=None, land_mask=None):
#         assert y is not None
#         assert land_mask is not None, "Land mask must be provided via model_kwargs"
#         with th.enable_grad():
#             x_in = x.detach().requires_grad_()
#             model_output = model(x_in, diffusion._scale_timesteps(t))
#             t_long = (t * diffusion.num_timesteps / 1000.0).long()
#             x_hat = diffusion._predict_xstart_from_eps(x_t=x_in, t=t_long, eps=model_output)
#             mask = ~th.isnan(y) #引导点的地方设置为 True
#             # visualize_tensor_map(
#             #     mask.float(), 
#             #     f"Guidance Mask (t={t_long[0].item()})",
#             #     os.path.join(vis_dir, f"mask_t{t_long[0].item():04d}.png"),
#             #     cmap='binary'
#             # )
#             diff = th.zeros_like(x_hat)
#             boundary_mask = create_boundary_mask(x.shape, border_width=1).to(x.device)
#             mask = mask & ~boundary_mask
#             diff[mask] = y[mask] - x_hat[mask]
#             # visualize_tensor_map(
#             #     diff,
#             #     f"Difference (y - x_hat) at t={t_long[0].item()}",
#             #     os.path.join(vis_dir, f"diff_t{t_long[0].item():04d}.png"),
#             #     cmap='RdBu_r',
#             #     vmin=-5,
#             #     vmax=5
#             # )
#             scalar_product = (diff.detach() * x_hat).sum() 
#             gradient = th.autograd.grad(scalar_product, x_in, retain_graph=False)[0] # retain_graph=False
#             print(f"Gradient Abs Sum: {th.abs(gradient).sum().item()}")
#             # visualize_tensor_map(
#             #     gradient,
#             #     f"Gradient before land mask (sum={th.abs(gradient).sum().item()})",
#             #     os.path.join(vis_dir, f"grad_before_t{t_long[0].item():04d}.png"),
#             #     cmap='RdBu_r'
#             # )
#             # visualize_tensor_map(
#             #     land_mask.float(), 
#             #     f"land Mask (t={t_long[0].item()})",
#             #     os.path.join(vis_dir, f"land_mask_t{t_long[0].item():04d}.png"),
#             #     cmap='binary'
#             # )
#             # gradient[land_mask] = 0.0
#             # visualize_tensor_map(
#             #     gradient,
#             #     f"Gradient after land mask (sum={th.abs(gradient).sum().item()})",
#             #     os.path.join(vis_dir, f"grad_after_t{t_long[0].item():04d}.png"),
#             #     cmap='RdBu_r'
#             # )
            
#             print(f"Gradient Abs Sum: {th.abs(gradient).sum().item()}")
            
#             # visualize_tensor_map(
#             #     gradient,
#             #     f"Gradient after boundary mask (sum={th.abs(gradient).sum().item()})",
#             #     os.path.join(vis_dir, f"grad_after_boundarymask_t{t_long[0].item():04d}.png"),
#             #     cmap='RdBu_r'
#             # )
#             gradient = th.nan_to_num(gradient, nan=0.0)
#             alphas_cumprod_sqrt = th.sqrt(_extract_into_tensor(diffusion.alphas_cumprod, t_long, x.shape))
#             print(f"Alphas_cumprod_sqrt: {alphas_cumprod_sqrt[0,0,0,0].item()}")
#         return gradient * args.grad_scale * alphas_cumprod_sqrt
#     return cond_fn




def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
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
        if not file.endswith(".nc"):
            continue

        path = os.path.join(args.sparse_data_path, file)
        try:
            with xr.open_dataset(path) as ds:
                var_name = 'thetao' if 'thetao' in ds.data_vars else 'temperature'
                if var_name not in ds.data_vars:
                    continue
                
                arr = ds[var_name].values.astype(np.float32).squeeze()
                mean_val = np.nanmean(arr)
                if mean_val > 100:
                    arr = arr - 273.15
                
                guided_arr_dict[file[:-3]] = arr
        except Exception as e:
            logger.log(f"Error processing {file}: {e}")

    logger.log("Successfully load the guided data!")

    softmask_kernel = get_gaussian_kernel(5, 1.0).float().to(dist_util.dev())

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    date = time.strftime("%m%d")
    config = f"s={args.grad_scale}_r={args.guided_rate}_loss={args.loss_model}_softmask={args.use_softmask}_sigma={args.use_sigma}"
    out_dir = os.path.join(os.environ.get("DIFFUSION_SAMPLE_LOGDIR", logger.get_dir()), date, config)
    os.makedirs(out_dir, exist_ok=True)

    logger.log("sampling...")
    loss_preds, loss_guideds, losses = [], [], []
    
    softmask_kernel = get_gaussian_kernel(5, 1.0).float().to("cuda")

    def cond_fn(x, t, p_mean_var, y=None):
        assert y is not None
        
        x = p_mean_var['pred_xstart']
        s = args.grad_scale
        gradient = 2 * (y - x)
        gradient = th.nan_to_num(gradient, nan=0.0)
        if args.use_softmask:
            size = softmask_kernel.shape[-1]
            gradient = F.conv2d(
                gradient,
                softmask_kernel.expand(gradient.shape[1], 1, size, size),
                stride=1,
                padding=size // 2,
                groups=gradient.shape[1]
            )
        return gradient * s
    for key, guided_arr in guided_arr_dict.items():
        all_images, all_loss_pred, all_loss_guided, all_loss = [], [], [], []
        logger.log(f"sampling {key}...")
        land_mask_np = np.isnan(guided_arr)
        
        while len(all_images) * args.batch_size < args.num_samples:
            guided_y, eval_y = split_guided_eval_batch_size(args.batch_size, guided_arr, args.guided_rate)
            land_mask_batch = th.from_numpy(
                np.stack([land_mask_np] * args.batch_size)
            ).to(dist_util.dev())
            
            model_kwargs = {
                "y": normalization(guided_y),
                # "land_mask": land_mask_batch
            }
            
            video_frames_dir = os.path.join(out_dir, f"{key}_video_frames_{int(time.time())}")
            os.makedirs(video_frames_dir, exist_ok=True)
            
            initial_noise = th.randn(
                (args.batch_size, args.in_channels, args.image_size_H, args.image_size_W),
                device=dist_util.dev()
            )
            visualize_and_save_frame(initial_noise, diffusion.num_timesteps - 1, video_frames_dir)
            
            sample_generator = diffusion.p_sample_loop_progressive(
                model_fn,
                (args.batch_size, args.in_channels, args.image_size_H, args.image_size_W),
                noise=initial_noise,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                # cond_fn=create_pigdm_cond_fn(model, diffusion, args),
                cond_fn=cond_fn,
                use_sigma=args.use_sigma,
                device=dist_util.dev(),
            )
            
            final_sample = None
            indices = list(range(diffusion.num_timesteps))[::-1]
            
            for i, result in enumerate(sample_generator):
                timestep = indices[i]
                final_sample = result["sample"]
                # if timestep > 0:
                #     visualize_and_save_frame(final_sample, timestep - 1, video_frames_dir)
            
            sample = final_sample
            sample = ((sample + 1) * 22.5 - 5).clamp(-5, 40)
            sample = sample.permute(0, 2, 3, 1).contiguous()
            
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            
            loss_pred = calculate_loss(sample, eval_y, args.loss_model)
            loss_guided = calculate_loss(sample, guided_y, args.loss_model)
            loss_total = (1 - args.guided_rate) * loss_pred + args.guided_rate * loss_guided
            
            all_loss_pred.append(loss_pred)
            all_loss_guided.append(loss_guided)
            all_loss.append(loss_total)
            all_images.extend([s.cpu().numpy() for s in gathered_samples])
            
            logger.log(f"created {len(all_images) * args.batch_size} samples")
            
            if args.num_samples <= args.batch_size:
                break
        
        arr = np.concatenate(all_images, axis=0)
        pred_loss = np.mean(all_loss_pred, axis=0)
        guided_loss = np.mean(all_loss_guided, axis=0)
        total_loss = np.mean(all_loss, axis=0)
        
        loss_preds.append(pred_loss)
        loss_guideds.append(guided_loss)
        losses.append(total_loss)
        
        if dist.get_rank() == 0:
            logger.log(f"Loss of {key} pred_loss: {pred_loss}, guided_loss: {guided_loss}, total_loss: {total_loss}")
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(out_dir, f"{key}_sample{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr)
            logger.log(f"sampling {key} complete")
        
        break  # For demo purposes

    dist.barrier()
    logger.log("Complete All Sample!")
    logger.log(f"Compute Total Loss: pred_loss: {np.mean(loss_preds, axis=0)}, guided_loss: {np.mean(loss_guideds, axis=0)}, total_loss: {np.mean(losses, axis=0)}")


def create_argparser():
    defaults = dict(
        image_size_H=173,
        image_size_W=360,
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        use_sigma=False,
        model_path="",
        sparse_data_path="",
        grad_scale=1.0,
        use_softmask=True,
        dynamic_guided=False,
        dynamic_guided_with_next=False,
        guided_rate=0.075,
        loss_model="l1",
        use_fp16=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.destroy_process_group()