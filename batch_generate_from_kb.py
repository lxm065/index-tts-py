from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

from indextts.infer_v2 import IndexTTS2

KB_PATH = Path(r"E:\vedio\cankao\knowledge_base.json")
SPK_DIR = Path(r"E:\vedio\cankao")
OUT_DIR = Path(r"E:\vedio\new")

TYPE_MAPPING = {
    "stay_core_selling_points": ("points", "selling"),
    "high_conversion_hooks": ("hooks", "conversion"),
    "create_anxiety_guidance": ("guidances", "anxiety"),
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def load_json_with_fallback(path: Path) -> dict:
    encodings = ["utf-8", "utf-8-sig", "gb18030"]
    last_error = None
    for enc in encodings:
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"Failed to read JSON from {path}: {last_error}")


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[\\/:*?\"<>|]", "_", name)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def mmss_to_compact(value: str) -> str:
    return value.replace(":", "")


def collect_items(data: dict) -> list[dict]:
    videos = (
        data.get("explosive_video_knowledge_base", {})
        .get("reference_video_list", {})
        .get("videos", [])
    )

    items: list[dict] = []
    for video in videos:
        file_name = video.get("file_name", {}).get("value", "unknown")
        video_name = sanitize_filename(Path(file_name).stem)

        for section_key, (list_key, section_alias) in TYPE_MAPPING.items():
            section = video.get(section_key, {})
            for point in section.get(list_key, []):
                text = (point.get("ai_optimized_text") or "").strip()
                if not text:
                    continue

                item = {
                    "video_name": video_name,
                    "section_alias": section_alias,
                    "id": point.get("id", 0),
                    "start": str(point.get("start", "00:00")),
                    "end": str(point.get("end", "00:00")),
                    "text": text,
                }
                items.append(item)

    return items


def build_output_path(item: dict) -> Path:
    file_name = (
        f"{item['video_name']}_"
        f"{item['section_alias']}_"
        f"{item['id']}_"
        f"{mmss_to_compact(item['start'])}-{mmss_to_compact(item['end'])}.mp3"
    )
    return OUT_DIR / file_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch TTS generation")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based index to start/resume from.",
    )
    parser.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="If GPU inference fails, switch to CPU and continue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"batch_generate_{ts}.log"
    fail_log_path = OUT_DIR / f"batch_failures_{ts}.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        print(f"日志文件: {log_path}")
        print(f"失败日志: {fail_log_path}")
        data = load_json_with_fallback(KB_PATH)
        items = collect_items(data)

        if not items:
            print("未读取到任何 ai_optimized_text，已退出。")
            return

        spk_files = sorted(SPK_DIR.glob("*.mp3"))
        if not spk_files:
            raise FileNotFoundError(f"未在 {SPK_DIR} 找到任何 .mp3 文件")

        print("读取到的 ai_optimized_text 预览：")
        for idx, item in enumerate(items, start=1):
            spk_file = spk_files[(idx - 1) % len(spk_files)]
            output_path = build_output_path(item)
            print(
                f"[{idx}] {item['video_name']} | {item['section_alias']} | "
                f"id={item['id']} | {item['start']}-{item['end']}"
            )
            print(f"spk_audio_prompt: {spk_file}")
            print(f"output_path: {output_path}")
            print(item["text"])
            print("-" * 80)

        print(f"总数：{len(items)}")
        answer = input("确认开始生成语音？(y/n): ").strip().lower()
        if answer != "y":
            print("已取消生成。")
            return

        tts = IndexTTS2(
            cfg_path=r"D:\AI\vits\index-tts\checkpoints\config.yaml",
            model_dir="checkpoints",
            use_fp16=False,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
        running_device = "gpu"

        for idx, (item, spk_file) in enumerate(zip(items, itertools.cycle(spk_files)), start=1):
            if idx < args.start_index:
                continue

            output_path = build_output_path(item)
            if output_path.exists():
                print(f"跳过已存在[{idx}]: {output_path.name}")
                continue

            print(f"生成中[{idx}/{len(items)}]: spk={spk_file.name} -> {output_path.name}")
            try:
                tts.infer(
                    spk_audio_prompt=str(spk_file),
                    text=item["text"],
                    output_path=str(output_path),
                    verbose=True,
                )
            except Exception as exc:
                err_text = traceback.format_exc()
                print(f"生成失败[{idx}]: {exc}")
                print(err_text)
                with fail_log_path.open("a", encoding="utf-8") as f:
                    f.write(f"index={idx}\n")
                    f.write(f"spk={spk_file}\n")
                    f.write(f"output={output_path}\n")
                    f.write(err_text)
                    f.write("\n" + "=" * 80 + "\n")

                if args.cpu_fallback and running_device != "cpu":
                    print("检测到异常，启用 CPU Fallback：切换到 CPU 重试当前条目。")
                    try:
                        del tts
                    except Exception:
                        pass

                    tts = IndexTTS2(
                        cfg_path=r"D:\AI\vits\index-tts\checkpoints\config.yaml",
                        model_dir="checkpoints",
                        use_fp16=False,
                        device="cpu",
                        use_cuda_kernel=False,
                        use_deepspeed=False,
                    )
                    running_device = "cpu"

                    try:
                        tts.infer(
                            spk_audio_prompt=str(spk_file),
                            text=item["text"],
                            output_path=str(output_path),
                            verbose=True,
                        )
                        print(f"CPU 重试成功[{idx}]，后续将继续使用 CPU。")
                        continue
                    except Exception as cpu_exc:
                        cpu_err_text = traceback.format_exc()
                        print(f"CPU 重试失败[{idx}]: {cpu_exc}")
                        print(cpu_err_text)
                        with fail_log_path.open("a", encoding="utf-8") as f:
                            f.write(f"cpu_retry_failed_index={idx}\n")
                            f.write(cpu_err_text)
                            f.write("\n" + "=" * 80 + "\n")

                print("建议重启后从断点继续：")
                print(f"uv run python batch_generate_from_kb.py --start-index {idx + 1}")
                return

        print("全部生成完成。")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
