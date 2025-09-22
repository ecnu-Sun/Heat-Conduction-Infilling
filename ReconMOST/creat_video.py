#!/usr/bin/env python3
import cv2
import os
import glob
from tqdm import tqdm

# --- 1. 配置参数 ---

# 包含图片帧的文件夹路径 (已为你填好)
IMAGE_FOLDER = '/hdd/SunZL/ReconMOST/testlog/0825/s=4.0_r=0.075_loss=l2_softmask=True_sigma=False/EN.4.2.2.f.analysis.g10.200211_video_frames_1756064868'

# 输出视频的文件名
VIDEO_NAME = 'diffusion_process_video.mp4'

# 视频的帧率 (Frames Per Second)
FPS = 30

# --- 2. 主程序 ---

def create_video_from_frames(image_folder, video_name, fps):
    """从文件夹中的图片帧创建视频。"""
    
    # 构造输出视频的完整路径
    output_path = os.path.join(image_folder, video_name)
    print(f"▶️  准备从文件夹创建视频:")
    print(f"    - 来源: {image_folder}")
    print(f"    - 输出: {output_path}")

    # 查找所有 frame_*.png 文件，并按数字顺序排序
    # glob.glob 会找到所有匹配的文件，sorted 会确保它们按 "frame_0000", "frame_0001"... 的顺序排列
    image_files = sorted(glob.glob(os.path.join(image_folder, 'frame_*.png')),reverse=True)

    if not image_files:
        print("\n❌ 错误: 在指定文件夹中未找到 'frame_*.png' 格式的图片。请检查路径是否正确。")
        return

    print(f"    - 找到 {len(image_files)} 帧图片。")

    # 读取第一张图片以获取视频的宽度和高度
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"\n❌ 错误: 无法读取第一帧图片: {image_files[0]}")
        return
        
    height, width, layers = first_frame.shape
    size = (width, height)
    print(f"    - 视频尺寸: {width}x{height}")

    # 初始化视频写入器 (VideoWriter)
    # 'mp4v' 是用于 .mp4 文件的编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    print("\n⏳ 正在逐帧写入视频...")
    # 使用 tqdm 显示进度条
    for image_file in tqdm(image_files, desc="处理帧"):
        frame = cv2.imread(image_file)
        video_writer.write(frame)

    # 释放资源
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"\n✅ 视频制作完成！已保存至: {output_path}")


if __name__ == '__main__':
    create_video_from_frames(IMAGE_FOLDER, VIDEO_NAME, FPS)