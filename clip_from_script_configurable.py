#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 script.json 的内容剪辑视频 (可选CPU/GPU加速)
使用方法：
  python clip_from_script_configurable.py          # 默认GPU模式
  python clip_from_script_configurable.py --cpu    # CPU模式
  python clip_from_script_configurable.py --gpu    # GPU模式
"""

import json
import os
import subprocess
import argparse
import time

# script.json 文件路径
SCRIPT_FILE = "script.json"

# 输出目录
OUTPUT_DIR = "qw"

# 源视频目录
SOURCE_DIR = "."

# 编码参数配置
ENCODE_PARAMS = {
    'gpu': {
        'name': 'GPU (NVIDIA NVENC)',
        'params': [
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda',
        ],
        'codec': [
            '-c:v', 'h264_nvenc',
            '-preset', 'p7',        # p7=最高质量
            '-tune', 'hq',
            '-rc', 'vbr',
            '-cq', '23',
            '-b:v', '0',
            '-c:a', 'aac',
            '-ar', '48000',         # 统一采样率 48kHz
            '-ac', '2',             # 统一双声道
            '-b:a', '192k'          # 提高音频码率
        ]
    },
    'cpu': {
        'name': 'CPU (libx264)',
        'params': [],
        'codec': [
            '-c:v', 'libx264',
            '-preset', 'fast',      # fast/medium/slow
            '-crf', '23',           # 恒定质量 (18=极高, 23=高)
            '-c:a', 'aac',
            '-ar', '48000',         # 统一采样率 48kHz
            '-ac', '2',             # 统一双声道
            '-b:a', '192k'          # 提高音频码率
        ]
    }
}


def load_script():
    """加载 script.json"""
    with open(SCRIPT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def clip_video(video_name, start, end, output_path, mode='gpu'):
    """
    剪辑视频片段
    :param video_name: 视频文件名（不含扩展名）
    :param start: 开始时间（秒）
    :param end: 结束时间（秒）
    :param output_path: 输出文件路径
    :param mode: 'gpu' 或 'cpu'
    """
    video_file = os.path.join(SOURCE_DIR, f"{video_name}.mp4")
    
    if not os.path.exists(video_file):
        print(f"⚠️  视频文件不存在：{video_file}")
        return False
    
    duration = end - start
    config = ENCODE_PARAMS[mode]
    
    cmd = ['ffmpeg', '-y']
    
    # 添加硬件加速参数（仅GPU模式）
    if mode == 'gpu':
        cmd.extend(config['params'])
    
    # 输入和时间参数
    cmd.extend([
        '-i', video_file,
        '-ss', str(start),
        '-t', str(duration),
    ])
    
    # 编码参数
    cmd.extend(config['codec'])
    
    # 输出
    cmd.extend([
        '-avoid_negative_ts', 'make_zero',
        output_path
    ])
    
    # 调试：打印FFmpeg命令（可选）
    if os.getenv('DEBUG_FFMPEG'):
        print(f"[DEBUG] FFmpeg命令: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        # 修复Windows编码问题：使用errors='ignore'忽略无法解码的字符
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, 
                               encoding='utf-8', errors='ignore')
        elapsed = time.time() - start_time
        print(f"✅ 已剪辑：{video_name} ({start}s-{end}s) -> {os.path.basename(output_path)} [耗时: {elapsed:.1f}s]")
        return True
    except subprocess.CalledProcessError as e:
        # 安全地获取错误信息
        error_msg = getattr(e, 'stderr', str(e))
        if error_msg:
            print(f"❌ 剪辑失败：{error_msg[:200]}")
        else:
            print(f"❌ 剪辑失败：{str(e)[:200]}")
        return False


def concat_videos(clip_files, output_path, mode='gpu'):
    """
    拼接所有视频片段
    :param clip_files: 视频文件路径列表
    :param output_path: 输出文件路径
    :param mode: 'gpu' 或 'cpu'
    
    注意：拼接时不使用硬件加速解码，只在编码时使用GPU
    """
    # 创建临时列表文件
    list_file = os.path.join(OUTPUT_DIR, "clips_list.txt")
    with open(list_file, 'w', encoding='utf-8') as f:
        for clip_file in clip_files:
            rel_path = os.path.basename(clip_file)
            f.write(f"file '{rel_path}'\n")
    
    config = ENCODE_PARAMS[mode]
    
    # 拼接命令：不使用硬件加速解码
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file]
    
    # 编码参数（GPU或CPU）- 保持与剪辑时完全一致的音频参数
    if mode == 'gpu':
        # GPU编码（不使用hwaccel解码，因为concat不支持）
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-preset', 'p7',
            '-tune', 'hq',
            '-rc', 'vbr',
            '-cq', '23',
            '-b:v', '0',
            '-c:a', 'aac',
            '-ar', '48000',        # 统一采样率
            '-ac', '2',            # 统一声道数
            '-b:a', '192k'         # 音频码率
        ])
    else:
        # CPU编码
        cmd.extend(config['codec'])
    
    # 输出
    cmd.append(output_path)
    
    try:
        start_time = time.time()
        # 修复Windows编码问题
        result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                               encoding='utf-8', errors='ignore')
        elapsed = time.time() - start_time
        print(f"✅ 已拼接：{output_path} [耗时: {elapsed:.1f}s]")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = getattr(e, 'stderr', str(e))
        if error_msg:
            print(f"❌ 拼接失败：{error_msg[:500]}")  # 增加到500字符看更多错误信息
        else:
            print(f"❌ 拼接失败：{str(e)[:500]}")
        return False


def check_gpu_support():
    """检查GPU编码器是否可用"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )
        if 'h264_nvenc' in result.stdout:
            return True
        return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description='视频剪辑脚本 (支持CPU/GPU加速)')
    parser.add_argument('--mode', choices=['cpu', 'gpu'], default='gpu',
                        help='编码模式: cpu 或 gpu (默认: gpu)')
    parser.add_argument('--cpu', action='store_true', help='使用CPU编码 (等同于 --mode cpu)')
    parser.add_argument('--gpu', action='store_true', help='使用GPU编码 (等同于 --mode gpu)')
    
    args = parser.parse_args()
    
    # 处理快捷参数
    if args.cpu:
        mode = 'cpu'
    elif args.gpu:
        mode = 'gpu'
    else:
        mode = args.mode
    
    # 检查GPU支持
    if mode == 'gpu':
        print("🔍 检查GPU编码器支持...")
        if check_gpu_support():
            print("✅ NVIDIA GPU编码器可用 (h264_nvenc)")
        else:
            print("⚠️  未检测到NVIDIA GPU编码器")
            print("💡 自动切换到CPU模式")
            mode = 'cpu'
    
    print(f"\n🎬 使用编码模式: {ENCODE_PARAMS[mode]['name']}\n")
    
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 加载剧本
    script = load_script()
    print(f"📺 已加载剧本：{SCRIPT_FILE}")
    
    # 记录总耗时
    total_start = time.time()
    
    # 按顺序处理所有片段
    clip_files = []
    clip_index = 0
    
    # 定义处理顺序
    order = ['停留', '观点', '亮点', '引导']
    
    for section in order:
        if section not in script:
            continue
        
        print(f"\n📝 处理片段类型：{section}")
        
        for item in script[section]:
            video_name = item.get('video', '')
            start = item.get('start', 0)
            end = item.get('end', 0)
            
            if not video_name or end <= start:
                continue
            
            # 生成输出文件名
            output_filename = f"clip_{clip_index:03d}_{section}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # 剪辑视频
            if clip_video(video_name, start, end, output_path, mode):
                clip_files.append(output_path)
                clip_index += 1
    
    if not clip_files:
        print("\n❌ 没有成功剪辑任何片段")
        return
    
    print(f"\n📋 共剪辑 {len(clip_files)} 个片段")
    
    # 验证所有片段的音频采样率
    print("\n🔍 验证音频采样率一致性...")
    sample_rates = {}
    for clip_file in clip_files[:5]:  # 只检查前5个片段
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=sample_rate',
                 '-of', 'default=noprint_wrappers=1:nokey=1', clip_file],
                capture_output=True, text=True, encoding='utf-8', errors='ignore'
            )
            if result.returncode == 0:
                rate = result.stdout.strip()
                sample_rates[os.path.basename(clip_file)] = rate
        except Exception:
            pass
    
    if sample_rates:
        unique_rates = set(sample_rates.values())
        if len(unique_rates) == 1:
            print(f"✅ 音频采样率统一: {list(unique_rates)[0]} Hz")
        else:
            print(f"⚠️  警告：检测到不同的采样率！")
            for file, rate in sample_rates.items():
                print(f"   {file}: {rate} Hz")
            print(f"💡 这可能导致拼接后变音，建议重新剪辑")
    
    # 拼接所有片段
    print("\n🔗 开始拼接所有片段...")
    final_output = os.path.join(OUTPUT_DIR, "final_output.mp4")
    if concat_videos(clip_files, final_output, mode):
        total_elapsed = time.time() - total_start
        
        print(f"\n🎉 视频剪辑完成：{final_output}")
        print(f"⏱️  总耗时：{total_elapsed:.1f} 秒")
        
        # 获取最终视频时长
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1', final_output],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            print(f"📺 视频时长：{duration:.2f} 秒")
            
            if mode == 'gpu':
                print(f"\n💡 使用GPU加速，相比CPU模式预计快3-10倍！")
    else:
        print("\n❌ 视频拼接失败")


if __name__ == "__main__":
    main()