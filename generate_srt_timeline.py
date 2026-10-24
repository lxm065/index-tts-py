import argparse
import importlib.util
import re
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torchaudio


TIMECODE_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


def parse_timecode(value: str) -> float:
    """Convert an SRT timecode into seconds."""

    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def parse_srt(srt_path: Path) -> list[dict[str, float | str]]:
    """Read SRT captions as ordered timeline entries."""

    content = srt_path.read_text(encoding="utf-8-sig")
    entries: list[dict[str, float | str]] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue

        timing_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue

        match = TIMECODE_PATTERN.search(lines[timing_index])
        if match is None:
            continue

        text = " ".join(lines[timing_index + 1 :]).strip()
        if not text:
            continue

        entries.append(
            {
                "start": parse_timecode(match.group("start")),
                "end": parse_timecode(match.group("end")),
                "text": text,
            }
        )

    return entries


def load_single_script(script_path: Path):
    """Load the configured single-sentence TTS script as a Python module."""

    spec = importlib.util.spec_from_file_location("single_tts_config", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load TTS script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_base_config(single_module, output_dir: Path):
    """Build a reusable TTS config from the single-sentence script defaults."""

    original_argv = list(sys.argv)
    try:
        sys.argv = [str(output_dir / "single-config"), "--text", "placeholder"]
        return single_module.parse_args()
    finally:
        sys.argv = original_argv


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load generated audio as a mono tensor."""

    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def generate_segments(single_module, base_config, entries, output_dir: Path) -> tuple[list[dict[str, float | str | Path]], int]:
    """Generate one WAV file for each subtitle entry with one shared TTS engine."""

    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    tts = single_module.build_tts(base_config)
    sample_rate = 22050
    generated = []

    for index, entry in enumerate(entries, start=1):
        segment_path = segments_dir / f"{index:03d}.wav"
        config = replace(base_config, text=str(entry["text"]), output_path=str(segment_path))
        print(f"[{index}/{len(entries)}] {entry['start']:.3f}s -> {entry['end']:.3f}s {entry['text']}", flush=True)
        tts.infer(
            spk_audio_prompt=config.spk_audio_prompt,
            text=config.text,
            output_path=config.output_path,
            verbose=False,
        )
        waveform, segment_sample_rate = load_audio(segment_path)
        sample_rate = segment_sample_rate
        generated.append(
            {
                **entry,
                "path": segment_path,
                "duration": waveform.shape[1] / segment_sample_rate,
            }
        )

    return generated, sample_rate


def compose_timeline(generated, output_path: Path, sample_rate: int, tail_padding: float) -> None:
    """Place generated segments on the subtitle timeline and save a full WAV."""

    end_time = max(float(item["end"]) for item in generated) + tail_padding
    total_frames = int(round(end_time * sample_rate))
    timeline = torch.zeros((1, total_frames), dtype=torch.float32)

    for item in generated:
        waveform, segment_sample_rate = load_audio(Path(item["path"]))
        if segment_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, segment_sample_rate, sample_rate)

        start_frame = int(round(float(item["start"]) * sample_rate))
        end_frame = min(start_frame + waveform.shape[1], timeline.shape[1])
        timeline[:, start_frame:end_frame] += waveform[:, : end_frame - start_frame]

    peak = timeline.abs().max()
    if peak > 1:
        timeline = timeline / peak * 0.98

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), timeline, sample_rate, encoding="PCM_S", bits_per_sample=16)


def write_report(generated, report_path: Path) -> None:
    """Write a concise timing report for the generated narration."""

    lines = ["index,start,end,slot_duration,audio_duration,overflow_seconds,text,path"]
    for index, item in enumerate(generated, start=1):
        slot_duration = float(item["end"]) - float(item["start"])
        audio_duration = float(item["duration"])
        overflow = max(0.0, audio_duration - slot_duration)
        text = str(item["text"]).replace('"', '""')
        lines.append(
            f'{index},{item["start"]:.3f},{item["end"]:.3f},{slot_duration:.3f},'
            f'{audio_duration:.3f},{overflow:.3f},"{text}","{item["path"]}"'
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command line options for SRT timeline generation."""

    parser = argparse.ArgumentParser(description="Generate a full TTS narration from an SRT timeline.")
    parser.add_argument("--srt-path", required=True)
    parser.add_argument("--single-script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="飞镖小乐园_完整解说.wav")
    parser.add_argument("--tail-padding", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Generate subtitle-aligned narration audio."""

    args = parse_args()
    srt_path = Path(args.srt_path)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name
    report_path = output_dir / "timing_report.csv"

    entries = parse_srt(srt_path)
    if not entries:
        raise RuntimeError(f"No subtitle entries found: {srt_path}")

    single_module = load_single_script(Path(args.single_script))
    base_config = build_base_config(single_module, output_dir)
    generated, sample_rate = generate_segments(single_module, base_config, entries, output_dir)
    compose_timeline(generated, output_path, sample_rate, args.tail_padding)
    write_report(generated, report_path)
    print(f"saved_audio={output_path}", flush=True)
    print(f"saved_report={report_path}", flush=True)


if __name__ == "__main__":
    main()
