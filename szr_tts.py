import argparse
import os
import sys
import subprocess

TARGET_VENV_PYTHON = r"D:\AI\vits\index-tts\.venv\Scripts\python.exe"
if os.path.exists(TARGET_VENV_PYTHON) and os.path.normcase(sys.executable) != os.path.normcase(TARGET_VENV_PYTHON):
    print(f"🔄 检测到当前在全局环境，正自动切换至专用虚拟环境: {TARGET_VENV_PYTHON}")
    try:
        sys.exit(subprocess.call([TARGET_VENV_PYTHON] + sys.argv))
    except KeyboardInterrupt:
        sys.exit(1)

# 临时添加项目根目录到 sys.path，解决在其他目录执行时找不到 indextts 模块的问题
sys.path.append(r"D:\AI\vits\index-tts")

# 引入 TTS 模型 (同 1-single.py)
from indextts.infer_v2 import IndexTTS2

def main():
    parser = argparse.ArgumentParser(description="IndexTTS2 命令行语音合成脚本")
    
    # 按照需求顺序设置位置参数
    parser.add_argument("ref_audio", type=str, help="参考音频路径 (例如: F:\\numhuman\\acai\\20260323_180804.wav)")
    parser.add_argument("text_file", type=str, help="参考文本路径 (例如: F:\\numhuman\\acai\\gfly\\tts.txt)")
    # 第三个参数设为可选的位置参数，如果不传则默认生成在 tts.txt 目录下
    parser.add_argument("output", type=str, nargs="?", default=None, help="输出音频文件路径 (可选，默认在文本同级目录下生成同名 .wav)")

    args = parser.parse_args()

    ref_audio = os.path.abspath(args.ref_audio)
    text_file = os.path.abspath(args.text_file)

    if not os.path.exists(ref_audio):
        print(f"错误: 参考音频不存在 -> {ref_audio}")
        sys.exit(1)

    if not os.path.exists(text_file):
        print(f"错误: 参考文本文件不存在 -> {text_file}")
        sys.exit(1)

    # 读取文本内容
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    except Exception as e:
        print(f"读取文本文件失败: {e}")
        sys.exit(1)

    if not text:
        print("错误: 文本文件内容为空！")
        sys.exit(1)

    # 处理输出文件路径：默认存在 tts.txt 目录下（替换后缀为 .wav），或者由用户指定
    output_file = args.output
    if not output_file:
        base_dir = os.path.dirname(text_file)
        # 将 tts.txt 替换为 tts.wav
        file_name = os.path.splitext(os.path.basename(text_file))[0]
        output_file = os.path.join(base_dir, f"{file_name}.wav")
    else:
        output_file = os.path.abspath(output_file)

    print("=========================================")
    print("TTS 语音合成任务信息")
    print(f"参考音频: {ref_audio}")
    print(f"参考文本: {text_file}")
    text_preview = text if len(text) < 50 else text[:50] + "..."
    print(f"合成文本: {text_preview}")
    print(f"输出路径: {output_file}")
    print("=========================================")
    print("正在加载 IndexTTS2 模型，请稍候...")

    # 保持与 1-single.py 相同的模型初始化逻辑
    try:
        tts = IndexTTS2(
            cfg_path=r"D:\AI\vits\index-tts\checkpoints\config.yaml",
            model_dir=r"D:\AI\vits\index-tts\checkpoints",
            use_fp16=False,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
    except Exception as e:
        print(f"模型初始化失败: {e}")
        sys.exit(1)

    print("模型加载完成，开始合成...")

    try:
        # 执行推理合成
        tts.infer(
            spk_audio_prompt=ref_audio,
            text=text,
            output_path=output_file,
            verbose=True
        )
        print(f"✅ 合成成功！音频已保存至: {output_file}")
    except Exception as e:
        print(f"❌ 合成过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
