from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path=r"D:\AI\vits\index-tts\checkpoints\config.yaml",
    model_dir="checkpoints",
    use_fp16=False,
    use_cuda_kernel=False,
    use_deepspeed=False,
)

text = "快跑起来！是他要来了！他要来抓我们了！"


tts.infer(spk_audio_prompt=r"C:\Users\Administrator\Downloads\120s.MP3", text=text, output_path="single.wav", verbose=True)
