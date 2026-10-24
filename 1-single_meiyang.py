import sys
sys.path.insert(0, r"D:\AI\vits\index-tts")
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path=r"D:\AI\vits\index-tts\checkpoints\config.yaml",
    model_dir="checkpoints",
    use_fp16=False,
    use_cuda_kernel=False,
    use_deepspeed=False,
)

text = "姐妹们别错过这件省心洗系列的新款条纹衬衫哦！它做的是带肌理的微透条纹设计，面料垂坠透气，穿上休闲又显瘦，特别百搭不挑下装，还支持机洗超好打理，夏天穿舒服又省心！"


tts.infer(spk_audio_prompt=r"E:\vedio\tts\60s.mp4", text=text, output_path="meiyang.wav", verbose=True)
