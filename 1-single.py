import argparse
import os
from dataclasses import dataclass

from indextts.infer_v2 import IndexTTS2


@dataclass(frozen=True)
class TtsConfig:
    """Store runtime options for one IndexTTS2 generation."""

    cfg_path: str
    model_dir: str
    spk_audio_prompt: str
    text: str
    output_path: str
    use_fp16: bool
    use_cuda_kernel: bool
    use_deepspeed: bool


def parse_args() -> TtsConfig:
    """Parse CLI arguments and environment fallbacks for single-file TTS."""

    parser = argparse.ArgumentParser(description="Generate one speech file with IndexTTS2.")
    parser.add_argument("--cfg-path", default=os.getenv("INDEXTTS_CFG_PATH", r"D:\AI\vits\index-tts\checkpoints\config.yaml"))
    parser.add_argument("--model-dir", default=os.getenv("INDEXTTS_MODEL_DIR", r"D:\AI\vits\index-tts\checkpoints"))
    parser.add_argument("--spk-audio-prompt", default=os.getenv("INDEXTTS_SPK_AUDIO", r"E:\vedio\tts\tts_samples\zh-CN-XiaoyiNeural.mp3"))
    parser.add_argument("--text", default=os.getenv("INDEXTTS_TEXT"), required=os.getenv("INDEXTTS_TEXT") is None)
    parser.add_argument("--output-path", default=os.getenv("INDEXTTS_OUTPUT", "single.wav"))
    parser.add_argument("--use-fp16", action="store_true", default=os.getenv("INDEXTTS_USE_FP16") == "1")
    parser.add_argument("--use-cuda-kernel", action="store_true", default=os.getenv("INDEXTTS_USE_CUDA_KERNEL") == "1")
    parser.add_argument("--use-deepspeed", action="store_true", default=os.getenv("INDEXTTS_USE_DEEPSPEED") == "1")
    args = parser.parse_args()

    return TtsConfig(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        spk_audio_prompt=args.spk_audio_prompt,
        text=args.text,
        output_path=args.output_path,
        use_fp16=args.use_fp16,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
    )


def build_tts(config: TtsConfig) -> IndexTTS2:
    """Create the IndexTTS2 engine from runtime configuration."""

    return IndexTTS2(
        cfg_path=config.cfg_path,
        model_dir=config.model_dir,
        use_fp16=config.use_fp16,
        use_cuda_kernel=config.use_cuda_kernel,
        use_deepspeed=config.use_deepspeed,
    )


def generate(config: TtsConfig) -> None:
    """Generate one audio file using the configured speaker prompt and text."""

    tts = build_tts(config)
    tts.infer(
        spk_audio_prompt=config.spk_audio_prompt,
        text=config.text,
        output_path=config.output_path,
        verbose=True,
    )


def main() -> None:
    """Run the single-sentence TTS command."""

    generate(parse_args())


if __name__ == "__main__":
    main()
