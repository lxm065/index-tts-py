"""
video_product_detector.py
使用本地 Ollama qwen2.5vl 模型定位视频中商品出现的时间窗口

用法:
    python video_product_detector.py <video_path> <product_name> [选项]

示例:
    python video_product_detector.py D:/videos/demo.mp4 "黑色T恤"
    python video_product_detector.py D:/videos/demo.mp4 "碎花长裙" --model qwen2.5vl:3b
    python video_product_detector.py D:/videos/demo.mp4 "白色帽子" --interval 3 --threshold 60
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import requests

# ─────────────────────────────────────────────
# 可调参数
# ─────────────────────────────────────────────
OLLAMA_URL          = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL       = "qwen2.5vl:7b"       # 视觉+文字一体模型
SAMPLE_INTERVAL_SEC = 4.0                  # 抽帧间隔（秒）
CONFIDENCE_THRESHOLD = 60                  # 命中分数阈值（0-100）
GAP_MERGE_SEC       = 12.0                  # 相邻命中帧合并间隔（秒）
MIN_WINDOW_SEC      = 4.0                  # 最短窗口时长（秒）
OLLAMA_TIMEOUT      = 30                   # 单次请求超时（秒），超时触发重启
OLLAMA_RESTART_WAIT = 5                    # 重启后等待秒数
OLLAMA_MAX_RETRIES  = 3                    # 超时后最多重试次数
JPEG_QUALITY        = 85                   # 帧图 JPEG 质量

# Ollama 路径配置
OLLAMA_EXE        = r"C:\Users\Administrator\AppData\Local\Programs\Ollama\ollama.exe"
OLLAMA_MODELS_DIR = r"D:\.ollama\models"


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def seconds_to_tc(sec: float) -> str:
    """秒数 → 时间码 HH:MM:SS.ss"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"[:-1]


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─────────────────────────────────────────────
# Ollama 调用（含自动重启）
# ─────────────────────────────────────────────

def restart_ollama(logger=None):
    """强制结束并重新启动本地 Ollama 服务（Windows）"""
    _log = (lambda msg: logger.warning(msg)) if logger else print
    _log("Ollama 无响应，正在重启...")
    try:
        subprocess.run("taskkill /F /IM ollama.exe",
                       shell=True, capture_output=True, timeout=10)
        subprocess.run('taskkill /F /IM "ollama app.exe"',
                       shell=True, capture_output=True, timeout=10)
        time.sleep(2)
        env = os.environ.copy()
        env["OLLAMA_MODELS"] = OLLAMA_MODELS_DIR
        subprocess.Popen(
            [OLLAMA_EXE, "serve"],
            env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
        )
        _log(f"等待 {OLLAMA_RESTART_WAIT}s 让 Ollama 重新就绪...")
        time.sleep(OLLAMA_RESTART_WAIT)
    except Exception as e:
        _log(f"重启失败: {e}")


def ollama_chat(model: str, prompt: str, image_path: str = None,
                logger=None) -> str:
    """
    调用 Ollama，支持图片输入。
    超时或连接失败时自动重启并重试，最多 OLLAMA_MAX_RETRIES 次。
    """
    msg = {"role": "user", "content": prompt}
    if image_path:
        msg["images"] = [encode_image_b64(image_path)]

    payload = {"model": model, "messages": [msg], "stream": False}

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            return (resp.json().get("message", {}).get("content") or "").strip()
        except requests.exceptions.Timeout:
            msg_log = f"[尝试 {attempt}/{OLLAMA_MAX_RETRIES}] 请求超时（>{OLLAMA_TIMEOUT}s）"
            if logger: logger.warning(msg_log)
            else: print(f"[WARN] {msg_log}")
            if attempt < OLLAMA_MAX_RETRIES:
                restart_ollama(logger)
            else:
                raise
        except requests.exceptions.ConnectionError:
            msg_log = f"[尝试 {attempt}/{OLLAMA_MAX_RETRIES}] 连接失败"
            if logger: logger.warning(msg_log)
            else: print(f"[WARN] {msg_log}")
            if attempt < OLLAMA_MAX_RETRIES:
                restart_ollama(logger)
            else:
                raise
    return ""


# ─────────────────────────────────────────────
# 核心：单帧分析
# ─────────────────────────────────────────────

def analyze_frame(image_path: str, product_name: str,
                  model: str, logger: logging.Logger) -> dict:
    """
    直接用 qwen2.5vl 对图片打分，返回 {score, reason}。
    使用中文 prompt，无需翻译，无需两阶段。
    """
    prompt = (
        f"请看这张图片，判断其中是否有【{product_name}】。\n"
        f"【{product_name}】可以是被穿着、手持、挂在衣架上、平铺展示——所有展示形式都算。\n\n"
        f"请只回复以下 JSON，不要有其他文字：\n"
        f"{{\"score\": 整数0到100, \"reason\": \"一句话说明\"}}\n\n"
        f"评分标准：0=完全没有，50=可能有但不确定，80=明显可见，100=清晰展示且是主体"
    )
    try:
        text = ollama_chat(model, prompt, image_path, logger)
        logger.debug("模型原始回复: %s", text[:150])

        # 解析 JSON（兼容模型在 JSON 前后夹杂文字）
        match = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = min(100, max(0, int(data.get("score", 0))))
            reason = data.get("reason", "")
            return {"score": score, "reason": reason}

        # 兜底：直接提取数字
        nums = re.findall(r'"score"\s*:\s*(\d+)', text)
        score = int(nums[0]) if nums else 0
        return {"score": score, "reason": text[:80]}

    except Exception as e:
        logger.error("帧分析失败: %s", e)
        return {"score": 0, "reason": f"error: {e}"}


# ─────────────────────────────────────────────
# 窗口合并
# ─────────────────────────────────────────────

def merge_hits_to_windows(hits: list) -> list:
    if not hits:
        return []

    windows = []
    cur_start  = hits[0]["time"]
    cur_end    = hits[0]["time"]
    cur_scores = [hits[0]["score"]]
    cur_reasons = [hits[0]["reason"]]

    for h in hits[1:]:
        if h["time"] - cur_end <= GAP_MERGE_SEC:
            cur_end = h["time"]
            cur_scores.append(h["score"])
            cur_reasons.append(h["reason"])
        else:
            windows.append(_make_window(cur_start, cur_end, cur_scores, cur_reasons))
            cur_start   = h["time"]
            cur_end     = h["time"]
            cur_scores  = [h["score"]]
            cur_reasons = [h["reason"]]

    windows.append(_make_window(cur_start, cur_end, cur_scores, cur_reasons))
    return [w for w in windows if w["duration"] >= MIN_WINDOW_SEC]


def _make_window(start, end, scores, reasons) -> dict:
    sample_reason = (reasons[len(reasons) // 2] or "")[:60]
    return {
        "start":             round(start, 2),
        "end":               round(end, 2),
        "start_tc":          seconds_to_tc(start),
        "end_tc":            seconds_to_tc(end),
        "duration":          round(end - start, 2),
        "avg_score":         round(sum(scores) / len(scores), 1),
        "dominant_category": sample_reason,
    }


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def process_video(video_path: str, product_name: str, keep_frames: bool,
                  sample_interval: float = SAMPLE_INTERVAL_SEC,
                  threshold: int = CONFIDENCE_THRESHOLD,
                  model: str = DEFAULT_MODEL):

    video_path = Path(video_path).resolve()
    if not video_path.exists():
        print(f"[ERROR] 视频文件不存在: {video_path}")
        sys.exit(1)

    video_stem = video_path.stem
    run_time   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = video_path.parent / run_time
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # 日志
    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger("detector")
    logger.info("视频: %s", video_path)
    logger.info("商品名: %s", product_name)
    logger.info("模型: %s", model)
    logger.info("输出目录: %s", output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("无法打开视频文件")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    step_frames  = max(1, int(fps * sample_interval))
    logger.info("FPS=%.2f  总帧数=%d  时长=%.1fs  抽帧间隔=%d帧",
                fps, total_frames, duration_sec, step_frames)

    hits        = []
    all_results = []
    frame_idx   = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        time_sec       = frame_idx / fps
        tc_str         = seconds_to_tc(time_sec).replace(":", "-")
        frame_filename = f"{video_stem}_{tc_str}.jpg"
        frame_path     = frames_dir / frame_filename

        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        logger.info("[%.1fs] 分析: %s", time_sec, frame_filename)
        result = analyze_frame(str(frame_path), product_name, model, logger)
        score  = result["score"]
        logger.info("  → score=%d  reason=%s", score, result["reason"][:60])

        all_results.append({
            "time":       round(time_sec, 3),
            "tc":         seconds_to_tc(time_sec),
            "frame_file": frame_filename,
            "score":      score,
            "reason":     result["reason"],
        })

        if score >= threshold:
            hits.append({
                "time":   time_sec,
                "score":  score,
                "reason": result["reason"],
            })

        if not keep_frames:
            try:
                os.remove(frame_path)
            except Exception:
                pass

        frame_idx += step_frames

    cap.release()
    logger.info("完成：共 %d 帧，命中 %d 帧", len(all_results), len(hits))

    # 保存 result.json
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "video":                video_stem,
            "video_path":           str(video_path),
            "product_name":         product_name,
            "model":                model,
            "run_time":             run_time,
            "total_frames_sampled": len(all_results),
            "hit_frames":           len(hits),
            "frames":               all_results,
        }, f, ensure_ascii=False, indent=2)

    # 合并并保存 result_simple.json
    windows     = merge_hits_to_windows(sorted(hits, key=lambda x: x["time"]))
    simple_path = output_dir / "result_simple.json"
    with open(simple_path, "w", encoding="utf-8") as f:
        json.dump({"video": video_stem, "suggested_windows": windows},
                  f, ensure_ascii=False, indent=2)

    logger.info("合并后窗口数: %d", len(windows))
    for w in windows:
        logger.info("  [%s → %s]  avg_score=%.1f", w["start_tc"], w["end_tc"], w["avg_score"])

    print("\n" + "="*60)
    print(f"✅ 分析完成  商品: {product_name}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 命中窗口数: {len(windows)}")
    for w in windows:
        print(f"   {w['start_tc']} → {w['end_tc']}  ({w['duration']}s)  avg={w['avg_score']}")
    print("="*60)

    return str(simple_path)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="使用 Ollama qwen2.5vl 定位视频商品时间窗口"
    )
    parser.add_argument("video_path",   help="源视频完整路径")
    parser.add_argument("product_name", help='商品名，例如："碎花裙" "黑色T恤"')
    parser.add_argument("--keep-frames",    action="store_true", default=True)
    parser.add_argument("--no-keep-frames", action="store_true", default=False,
                        help="分析后删除抽帧 jpg")
    parser.add_argument("--interval",  type=float, default=SAMPLE_INTERVAL_SEC,
                        help=f"抽帧间隔秒数（默认 {SAMPLE_INTERVAL_SEC}）")
    parser.add_argument("--threshold", type=int,   default=CONFIDENCE_THRESHOLD,
                        help=f"命中分数阈值（默认 {CONFIDENCE_THRESHOLD}）")
    parser.add_argument("--model",     default=DEFAULT_MODEL,
                        help=f"使用的视觉模型（默认 {DEFAULT_MODEL}，也可用 qwen2.5vl:3b）")
    args = parser.parse_args()

    keep = args.keep_frames and not args.no_keep_frames
    process_video(
        args.video_path, args.product_name, keep,
        args.interval, args.threshold, args.model
    )


if __name__ == "__main__":
    main()