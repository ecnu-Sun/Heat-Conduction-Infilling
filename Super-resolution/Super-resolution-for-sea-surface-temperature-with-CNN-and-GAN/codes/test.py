import os
import os.path as osp
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict
import math

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

# ==============================================================================
# --- 您可以在这里定义需要测试的特定日期 ---
# --- 如果列表为空，则会测试数据集中的所有文件 ---
# ==============================================================================
# TARGET_DATES = [
#     # --- 核心对比组 (冬季)：强厄尔尼诺 vs 中性 ---
#     "19830115",  # 强厄尔尼诺事件的峰值期 (冬季)
#     "19840115",  # 中性状态 (冬季)

#     # --- 新增对比组 (夏季)：厄尔尼诺发展 vs 中性 ---
#     "19820815",  # 强厄尔尼诺事件的快速发展期 (夏季)
#     "19840815",  # 中性状态 (夏季)

#     # --- 新增补充样本：峰值前期 和 衰退期 ---
#     "19821215",  # 厄尔尼诺事件达到顶峰前的一个月
#     "19830815",  # 厄尔尼诺事件开始衰退后的夏季
# ]
TARGET_DATES =[]
# ==============================================================================

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # 从配置文件读取掩码路径
    static_mask = None
    if opt['datasets']['test_1'] and opt['datasets']['test_1'].get('dataroot_mask', None):
        mask_path = opt['datasets']['test_1']['dataroot_mask']
        if os.path.exists(mask_path):
            logger.info(f"正在加载静态海洋掩码: {mask_path}")
            static_mask = np.load(mask_path)
        else:
            logger.error(f"静态掩码文件未找到: {mask_path}")
    else:
        logger.warning("未在测试集配置中找到掩码路径(dataroot_mask)，将不使用外部掩码。")

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)

        # --- 根据 TARGET_DATES 过滤数据集 ---
        if TARGET_DATES:
            logger.info(f"检测到目标日期列表，将只测试以下日期: {TARGET_DATES}")
            original_gt_count = len(test_set.paths_GT)
            test_set.paths_GT = [path for path in test_set.paths_GT if osp.splitext(osp.basename(path))[0] in TARGET_DATES]
            if test_set.paths_LQ:
                test_set.paths_LQ = [path for path in test_set.paths_LQ if osp.splitext(osp.basename(path))[0] in TARGET_DATES]
            logger.info(f"文件列表已过滤。GT文件数从 {original_gt_count} 减少到 {len(test_set.paths_GT)}.")
        
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('最终测试的图像数量 in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    #### create model
    model = create_model(opt)

    #### testing loop
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['mse'] = []
        test_results['ssim'] = []
        
        # --- 关键修改：定义与 train.py 一致的换算因子 ---
        CONVERSION_FACTOR = 45.0
        
        for data in test_loader:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
            model.feed_data(data, need_GT=need_GT)
            img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            model.test()
            visuals = model.get_current_visuals(need_GT=need_GT)
            sr_img = util.tensor2img(visuals['SR'])

            # save images
            # suffix = opt['suffix']
            # save_img_path = osp.join(dataset_dir, f"{img_name}{suffix or ''}.npy")
            # util.save_img(sr_img, save_img_path)

            if need_GT:
                gt_img = util.tensor2img(visuals['GT'])
                crop_border = opt['scale']

                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border]

                cropped_mask = None
                if static_mask is not None:
                    cropped_mask = static_mask[crop_border:-crop_border, crop_border:-crop_border]
                
                # --- 核心计算逻辑对齐 ---
                current_mse = util.calculate_mse(cropped_sr_img, cropped_gt_img, cropped_mask)
                test_results['mse'].append(current_mse)
                
                current_ssim = util.calculate_ssim(cropped_sr_img, cropped_gt_img)
                test_results['ssim'].append(current_ssim)

                # --- 关键修改：逐行日志也使用物理单位RMSE ---
                if current_mse > 0:
                    current_psnr = 20 * math.log10(255.0 / math.sqrt(current_mse))
                    current_rmse_pixel = math.sqrt(current_mse)
                    current_rmse_physical = current_rmse_pixel / 255. * CONVERSION_FACTOR
                else:
                    current_psnr = float('inf')
                    current_rmse_physical = 0.0
                
                logger.info('{:20s} - RMSE: {:.6f}; PSNR: {:.6f} dB; SSIM: {:.6f};'.format(
                    img_name, current_rmse_physical, current_psnr, current_ssim))

        if need_GT and test_results['mse']:
            # --- 最终结果的聚合与报告逻辑，与train.py完全对齐 ---
            ave_mse = sum(test_results['mse']) / len(test_results['mse'])
            
            if ave_mse <= 0:
                ave_rmse_pixel = 0.0
                ave_psnr = float('inf')
            else:
                ave_rmse_pixel = math.sqrt(ave_mse)
                ave_psnr = 20 * math.log10(255.0 / ave_rmse_pixel)

            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            final_ave_rmse_physical = ave_rmse_pixel / 255. * CONVERSION_FACTOR
            
            logger.info('----Average results for {}----'.format(test_set_name))
            logger.info('MSE: {:.4e}'.format(ave_mse))
            logger.info('PSNR: {:.6f} dB'.format(ave_psnr))
            logger.info('SSIM: {:.6f}'.format(ave_ssim))
            logger.info('RMSE (physical unit, factor={}): {:.6f}'.format(CONVERSION_FACTOR, final_ave_rmse_physical))

if __name__ == '__main__':
    main()