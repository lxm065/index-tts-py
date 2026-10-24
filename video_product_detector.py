"""
video_product_detector.py
使用本地 Ollama moondream 模型定位视频中商品出现的时间窗口

识别流程（两阶段）：
  Stage 1 - moondream  : 视觉描述，输出图片中服饰的自然语言描述
  Stage 2 - 文字模型    : 语义判断描述是否匹配商品名（支持 qwen2.5/llama3/gemma 等）
  Fallback - 关键词匹配 : 若文字模型不可用，退回关键词匹配打分

用法:
    python video_product_detector.py <video_path> <product_name> [选项]

示例:
    python video_product_detector.py D:/videos/demo.mp4 "碎花裙"
    python video_product_detector.py D:/videos/demo.mp4 "黑色T恤" --text-model qwen2.5
    python video_product_detector.py D:/videos/demo.mp4 "长裙子" --interval 3 --threshold 60
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import requests

# ─────────────────────────────────────────────
# 可调参数
# ─────────────────────────────────────────────
OLLAMA_URL        = "http://127.0.0.1:11434/api/chat"
VISION_MODEL      = "moondream"          # 视觉描述模型（必须支持图片）
TEXT_MODEL        = "qwen2.5"            # 语义判断模型（纯文本，可改 llama3/gemma2 等）
SAMPLE_INTERVAL_SEC = 2.0               # 抽帧间隔（秒）
CONFIDENCE_THRESHOLD = 60               # 命中分数阈值（0-100）
GAP_MERGE_SEC     = 5.0                 # 相邻命中帧合并间隔（秒）
MIN_WINDOW_SEC    = 2.0                 # 最短窗口时长（秒）
OLLAMA_TIMEOUT    = 120                 # 请求超时（秒）
JPEG_QUALITY      = 85                  # 帧图 JPEG 质量


# ─────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────

def seconds_to_tc(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"[:-1]


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ollama_chat(model: str, messages: list, image_b64: str = None, timeout: int = OLLAMA_TIMEOUT) -> str:
    """通用 Ollama /api/chat 调用，返回模型文本或空串"""
    if image_b64:
        messages[-1]["images"] = [image_b64]
    payload = {"model": model, "messages": messages, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return (resp.json().get("message", {}).get("content") or "").strip()


def check_model_available(model: str, logger: logging.Logger = None) -> bool:
    """
    检查 Ollama 中某模型是否已拉取。
    若 /api/tags 无法访问（网络异常），默认返回 True 让实际调用来验证。
    """
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        r.raise_for_status()
        model_base = model.split(":")[0]
        for m in r.json().get("models", []):
            raw = m.get("name", "")
            raw_base = raw.split(":")[0]
            if raw == model or raw_base == model or raw == model_base or raw_base == model_base:
                return True
        # 能查到列表但没找到 → 真的没装
        if logger:
            logger.warning("模型 [%s] 不在 ollama list 中，将使用关键词兜底", model)
        return False
    except Exception:
        # 连不上 /api/tags（常见于开发环境隔离），乐观假设模型存在，让调用失败时再降级
        if logger:
            logger.info("无法查询 ollama 模型列表，假设 [%s] 可用，调用失败会自动降级", model)
        return True


# ─────────────────────────────────────────────
# Stage 1：moondream 视觉描述
# ─────────────────────────────────────────────

def describe_frame(image_path: str, logger: logging.Logger) -> str:
    """让 moondream 用自然语言描述图片中的服饰"""
    b64 = encode_image_b64(image_path)
    prompt = (
        "Describe the clothing items visible in this image in detail. "
        "Include: colors, patterns (solid/striped/floral/plaid/etc.), "
        "garment types, length, and any notable features."
    )
    try:
        text = ollama_chat(VISION_MODEL, [{"role": "user", "content": prompt}], b64)
        if not text:
            logger.warning("moondream 返回空内容")
        return text
    except requests.exceptions.ConnectionError:
        logger.error("无法连接到 Ollama，请确认服务已启动")
        return ""
    except Exception as e:
        logger.error("视觉描述失败: %s", e)
        return ""


# ─────────────────────────────────────────────
# Stage 2：文字模型语义判断
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Fallback：关键词匹配打分（通用扩展版）
# ─────────────────────────────────────────────

# 颜色 / 花纹 / 品类 三类映射，覆盖常见中文商品名
_COLOR_MAP = {
    "黑": ["black", "dark"],
    "白": ["white", "cream", "ivory"],
    "红": ["red", "crimson"],
    "蓝": ["blue", "navy", "indigo"],
    "绿": ["green", "olive"],
    "黄": ["yellow", "golden"],
    "灰": ["gray", "grey"],
    "粉": ["pink", "rose"],
    "紫": ["purple", "violet", "lavender"],
    "棕": ["brown", "tan", "camel"],
    "橙": ["orange"],
    "米": ["beige", "cream", "khaki"],
    "卡其": ["khaki", "beige"],
}

_PATTERN_MAP = {
    "碎花": ["floral", "flower", "floral pattern", "flower print"],
    "花": ["floral", "flower"],
    "格子": ["plaid", "checkered", "gingham", "tartan"],
    "条纹": ["striped", "stripes"],
    "印花": ["printed", "print", "pattern"],
    "纯色": ["solid", "plain"],
    "豹纹": ["leopard", "animal print"],
    "迷彩": ["camo", "camouflage"],
    "波点": ["polka dot", "dotted"],
}

_CATEGORY_MAP = {
    "t恤": ["t-shirt", "tshirt", "tee", "shirt"],
    "衬衫": ["shirt", "blouse", "button-up"],
    "外套": ["jacket", "coat", "outerwear"],
    "卫衣": ["hoodie", "sweatshirt"],
    "裤子": ["pants", "trousers"],
    "牛仔裤": ["jeans", "denim pants"],
    "牛仔": ["jeans", "denim"],
    "长裙": ["long skirt", "maxi skirt", "long dress"],
    "短裙": ["short skirt", "mini skirt"],
    "连衣裙": ["dress", "gown"],
    "裙": ["skirt", "dress"],
    "帽子": ["hat", "cap"],
    "帽": ["hat", "cap"],
    "包": ["bag", "handbag", "purse"],
    "鞋": ["shoes", "sneakers", "boots"],
    "袜": ["socks"],
    "毛衣": ["sweater", "knitwear", "knit"],
    "针织": ["knit", "knitwear"],
    "大衣": ["coat", "overcoat"],
    "风衣": ["trench coat", "windbreaker"],
    "背心": ["vest", "tank top"],
    "吊带": ["camisole", "spaghetti strap"],
    "羽绒": ["down jacket", "puffer jacket"],
    "毛绒": ["fleece", "fuzzy", "fluffy", "plush"],
    "皮衣": ["leather jacket", "leather coat"],
    "西装": ["suit", "blazer"],
}



def translate_product_name(product_name: str) -> str:
    """
    把中文商品名翻译成英文描述短语，供英文模型理解。
    例如: "黑色T恤" → "black t-shirt"
         "碎花长裙" → "floral long skirt"
    """
    name_lower = product_name.lower()  # 统一小写，解决 T恤/t恤 大小写不一致
    colors   = []
    patterns = []
    cats     = []

    for zh, en_list in _COLOR_MAP.items():
        if zh in name_lower:
            colors.append(en_list[0])

    for zh, en_list in _PATTERN_MAP.items():
        if zh in name_lower:
            patterns.append(en_list[0])

    for zh, en_list in _CATEGORY_MAP.items():
        if zh in name_lower:
            cats.append(en_list[0])
            break  # 只取第一个匹配的品类，避免重复

    parts = list(dict.fromkeys(colors + patterns + cats))  # 保序去重
    return " ".join(parts) if parts else product_name




def semantic_score(description: str, product_name: str, logger: logging.Logger, model: str = TEXT_MODEL) -> int:
    """
    用纯文字 LLM 判断 description 是否包含 product_name 描述的商品。
    返回 0-100 分数。
    注意：prompt 使用纯英文，避免中文在某些模型中乱码导致误判。
    """
    # 将商品名转为英文描述，防止中文乱码
    product_en = translate_product_name(product_name)
    logger.debug("商品名英文化: %s -> %s", product_name, product_en)

    prompt = (
        f"Image description: \"{description}\"\n\n"
        f"Is a {product_en} visible in this image?\n"
        f"Consider: worn on body, held in hand, on a hanger, displayed, or lying flat — all count.\n\n"
        f"Answer with a JSON object containing:\n"
        f"- score: integer 0 to 100 (0=not present, 100=clearly visible)\n"
        f"- reason: one short sentence explaining your answer\n\n"
        f"JSON answer:"
    )
    try:
        text = ollama_chat(model, [{"role": "user", "content": prompt}])
        logger.debug("语义模型原始回复: %s", text[:120])
        # 提取 JSON（兼容模型在 JSON 前后加多余文字的情况）
        match = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return min(100, max(0, int(data.get("score", 0))))
        # 兜底：直接找数字
        nums = re.findall(r'"score"\s*:\s*(\d+)', text)
        if nums:
            return int(nums[0])
        nums = re.findall(r'\b(\d{1,3})\b', text)
        return int(nums[0]) if nums else 0
    except Exception as e:
        logger.debug("语义判断失败（将使用关键词兜底）: %s", e)
        return -1  # -1 表示失败，触发 fallback


def keyword_score(description: str, product_name: str) -> int:
    """关键词匹配打分（兜底方案）"""
    text = description.lower()
    keywords = []

    for zh, en_list in _COLOR_MAP.items():
        if zh in product_name:
            keywords.extend(en_list)
    for zh, en_list in _PATTERN_MAP.items():
        if zh in product_name:
            keywords.extend(en_list)
    for zh, en_list in _CATEGORY_MAP.items():
        if zh in product_name:
            keywords.extend(en_list)

    if not keywords:
        return 0

    keywords = list(dict.fromkeys(keywords))
    hits = [kw for kw in keywords if kw in text]

    if not hits:
        return 0
    ratio = len(hits) / len(keywords)
    return int(40 + 60 * ratio)


# ─────────────────────────────────────────────
# 统一入口：分析单帧
# ─────────────────────────────────────────────

def analyze_frame(image_path: str, product_name: str,
                  use_text_model: bool, logger: logging.Logger,
                  text_model: str = TEXT_MODEL) -> dict:
    """
    返回 {score, description, method}
      method: "semantic" | "keyword" | "empty"
    """
    # Stage 1: 视觉描述
    description = describe_frame(image_path, logger)
    if not description:
        return {"score": 0, "description": "", "method": "empty"}

    score = -1
    method = "keyword"

    # Stage 2: 语义判断（若文字模型可用）
    if use_text_model:
        score = semantic_score(description, product_name, logger, model=text_model)
        if score >= 0:
            method = "semantic"

    # Fallback: 关键词匹配
    if score < 0:
        score = keyword_score(description, product_name)
        method = "keyword"

    return {"score": score, "description": description, "method": method}


# ─────────────────────────────────────────────
# 窗口合并
# ─────────────────────────────────────────────

def merge_hits_to_windows(hits: list) -> list:
    if not hits:
        return []
    windows = []
    cur_start = hits[0]["time"]
    cur_end   = hits[0]["time"]
    cur_scores = [hits[0]["score"]]
    cur_descs  = [hits[0]["description"]]

    for h in hits[1:]:
        if h["time"] - cur_end <= GAP_MERGE_SEC:
            cur_end = h["time"]
            cur_scores.append(h["score"])
            cur_descs.append(h["description"])
        else:
            windows.append(_make_window(cur_start, cur_end, cur_scores, cur_descs))
            cur_start  = h["time"]
            cur_end    = h["time"]
            cur_scores = [h["score"]]
            cur_descs  = [h["description"]]

    windows.append(_make_window(cur_start, cur_end, cur_scores, cur_descs))
    return [w for w in windows if w["duration"] >= MIN_WINDOW_SEC]


def _make_window(start, end, scores, descs) -> dict:
    # dominant_category：取所有描述的前40字拼接后截断作为代表
    sample_desc = (descs[len(descs)//2][:60] if descs else "").replace("\n", " ")
    return {
        "start":              round(start, 2),
        "end":                round(end, 2),
        "start_tc":           seconds_to_tc(start),
        "end_tc":             seconds_to_tc(end),
        "duration":           round(end - start, 2),
        "avg_score":          round(sum(scores) / len(scores), 1),
        "dominant_category":  sample_desc,
    }


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def process_video(video_path: str, product_name: str, keep_frames: bool,
                  sample_interval: float = SAMPLE_INTERVAL_SEC,
                  threshold: int = CONFIDENCE_THRESHOLD,
                  text_model: str = TEXT_MODEL):

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

    # 检查文字模型是否可用
    use_text_model = check_model_available(text_model, logger)
    if use_text_model:
        logger.info("✅ 文字模型 [%s] 可用，启用两阶段语义识别", text_model)
    else:
        logger.warning("⚠️  文字模型 [%s] 不可用，将使用关键词匹配兜底", text_model)

    logger.info("视频: %s", video_path)
    logger.info("商品名: %s", product_name)
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
        result = analyze_frame(str(frame_path), product_name, use_text_model, logger, text_model=text_model)
        score  = result["score"]
        logger.info("  → score=%d  method=%s  desc=%s",
                    score, result["method"], result["description"][:60])

        all_results.append({
            "time":        round(time_sec, 3),
            "tc":          seconds_to_tc(time_sec),
            "frame_file":  frame_filename,
            "score":       score,
            "method":      result["method"],
            "description": result["description"],
        })

        if score >= threshold:
            hits.append({
                "time":        time_sec,
                "score":       score,
                "description": result["description"][:60],
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
            "run_time":             run_time,
            "text_model_used":      text_model if use_text_model else None,
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
        description="使用 Ollama moondream + 文字模型定位视频商品时间窗口"
    )
    parser.add_argument("video_path",    help="源视频完整路径")
    parser.add_argument("product_name",  help='商品名，例如："碎花裙" "黑色T恤"')
    parser.add_argument("--keep-frames", action="store_true", default=True)
    parser.add_argument("--no-keep-frames", action="store_true", default=False)
    parser.add_argument("--interval",    type=float, default=SAMPLE_INTERVAL_SEC,
                        help=f"抽帧间隔秒数（默认 {SAMPLE_INTERVAL_SEC}）")
    parser.add_argument("--threshold",   type=int,   default=CONFIDENCE_THRESHOLD,
                        help=f"命中分数阈值（默认 {CONFIDENCE_THRESHOLD}）")
    parser.add_argument("--text-model",  default=TEXT_MODEL,
                        help=f"语义判断用的纯文字模型（默认 {TEXT_MODEL}，可改 llama3/gemma2 等）")
    args = parser.parse_args()

    keep = args.keep_frames and not args.no_keep_frames
    process_video(
        args.video_path, args.product_name, keep,
        args.interval, args.threshold, args.text_model
    )


if __name__ == "__main__":
    main()


import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests

# ─────────────────────────────────────────────
# 可调参数
# ─────────────────────────────────────────────
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "moondream"
SAMPLE_INTERVAL_SEC = 2.0    # 每隔多少秒抽一帧
CONFIDENCE_THRESHOLD = 50    # 分数 >= 此值视为"命中"
GAP_MERGE_SEC = 5.0          # 相邻命中帧间隔 <= 此值合并为同一窗口
MIN_WINDOW_SEC = 2.0         # 窗口最短时长（过滤噪声）
OLLAMA_TIMEOUT = 120         # 单帧请求超时秒数
JPEG_QUALITY = 85            # 保存帧的 JPEG 质量


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def seconds_to_tc(sec: float) -> str:
    """秒数 → 时间码 HH:MM:SS.ss"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"[:-1]  # 保留两位小数


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_keyword_list(product_name: str) -> list:
    """
    从商品名提取中英文关键词，用于匹配模型自由描述。
    例如 "黑色T恤" → ["black", "dark", "黑", "t-shirt", "tshirt", "tee", "shirt", "t恤"]
    """
    keywords = []
    color_map = {
        "黑": ["black", "dark"],
        "白": ["white"],
        "红": ["red"],
        "蓝": ["blue", "navy"],
        "绿": ["green"],
        "黄": ["yellow"],
        "灰": ["gray", "grey"],
        "粉": ["pink"],
        "紫": ["purple"],
        "棕": ["brown"],
        "橙": ["orange"],
        "米": ["beige", "cream"],
    }
    category_map = {
        "t恤": ["t-shirt", "tshirt", "tee", "shirt"],
        "衬衫": ["shirt", "blouse"],
        "外套": ["jacket", "coat", "outerwear"],
        "卫衣": ["hoodie", "sweatshirt"],
        "裤子": ["pants", "trousers"],
        "牛仔": ["jeans", "denim"],
        "裙": ["skirt", "dress"],
        "帽子": ["hat", "cap"],
        "帽":   ["hat", "cap"],
        "包":   ["bag", "handbag", "purse"],
        "鞋":   ["shoes", "sneakers", "boots"],
        "袜":   ["socks"],
        "毛衣": ["sweater", "knit"],
        "大衣": ["coat", "overcoat"],
        "背心": ["vest", "tank top", "tank"],
        "羽绒": ["down jacket", "puffer"],
        "毛绒": ["fleece", "fuzzy", "fluffy", "plush"],
    }
    for zh_color, en_colors in color_map.items():
        if zh_color in product_name:
            keywords.extend(en_colors)
            keywords.append(zh_color)
    for zh_cat, en_cats in category_map.items():
        if zh_cat in product_name:
            keywords.extend(en_cats)
            keywords.append(zh_cat)
    # 原始名也加入
    keywords.append(product_name.lower())
    return list(dict.fromkeys(keywords))


def build_prompt(product_name: str) -> str:
    # moondream 轻量模型不擅长遵循结构化格式，用简单描述问句更稳定
    # 打分逻辑由 parse_response 中的关键词匹配完成
    return "Describe all clothing items visible in this image. Include their colors and types."


def query_moondream(image_path: str, product_name: str, logger: logging.Logger) -> dict:
    """调用 Ollama moondream，返回 {score, category, reason}"""
    b64 = encode_image_b64(image_path)
    prompt = build_prompt(product_name)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64]
            }
        ],
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}).get("content") or "").strip()
        if not text:
            logger.warning("模型返回空内容，跳过此帧")
            return {"score": 0, "category": "empty_response", "reason": "model returned empty"}
        logger.debug("Raw response: %s", text)
        return parse_response(text, product_name)
    except requests.exceptions.ConnectionError:
        logger.error("无法连接到 Ollama，请确认服务已启动（ollama serve）")
        return {"score": 0, "category": "error", "reason": "connection error"}
    except Exception as e:
        logger.error("查询失败: %s", e)
        return {"score": 0, "category": "error", "reason": str(e)}


def parse_response(text: str, product_name: str) -> dict:
    """
    moondream 返回自由描述文本，通过关键词匹配打分。
    命中关键词越多、权重越高，分数越高（最高100）。
    """
    text_lower = text.lower()
    keywords = build_keyword_list(product_name)

    hit_keywords = [kw for kw in keywords if kw.lower() in text_lower]
    if not keywords:
        score = 0
    else:
        # 命中比例 * 100，至少命中1个给基础分40，全中100
        ratio = len(hit_keywords) / len(keywords)
        score = int(40 + 60 * ratio) if hit_keywords else 0

    # 提取 category：取文本前80字符作为简短描述
    category = text[:80].replace("\n", " ").strip() if text else "unknown"

    return {
        "score": score,
        "category": category,
        "reason": f"Keywords matched: {hit_keywords}" if hit_keywords else "No matching keywords"
    }


# ─────────────────────────────────────────────
# 窗口合并
# ─────────────────────────────────────────────

def merge_hits_to_windows(hits: list[dict]) -> list[dict]:
    """
    hits: [{time, score, category}, ...]（已按 time 排序）
    返回合并后的窗口列表
    """
    if not hits:
        return []

    windows = []
    cur_start = hits[0]["time"]
    cur_end = hits[0]["time"]
    cur_scores = [hits[0]["score"]]
    cur_cats = [hits[0]["category"]]

    for h in hits[1:]:
        if h["time"] - cur_end <= GAP_MERGE_SEC:
            cur_end = h["time"]
            cur_scores.append(h["score"])
            cur_cats.append(h["category"])
        else:
            windows.append(_make_window(cur_start, cur_end, cur_scores, cur_cats))
            cur_start = h["time"]
            cur_end = h["time"]
            cur_scores = [h["score"]]
            cur_cats = [h["category"]]

    windows.append(_make_window(cur_start, cur_end, cur_scores, cur_cats))

    # 过滤太短的窗口
    windows = [w for w in windows if w["duration"] >= MIN_WINDOW_SEC]
    return windows


def _make_window(start, end, scores, cats) -> dict:
    from collections import Counter
    dominant = Counter(cats).most_common(1)[0][0]
    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "start_tc": seconds_to_tc(start),
        "end_tc": seconds_to_tc(end),
        "duration": round(end - start, 2),
        "avg_score": round(sum(scores) / len(scores), 1),
        "dominant_category": dominant
    }


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def process_video(video_path: str, product_name: str, keep_frames: bool,
                  sample_interval: float = SAMPLE_INTERVAL_SEC,
                  threshold: int = CONFIDENCE_THRESHOLD):
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        print(f"[ERROR] 视频文件不存在: {video_path}")
        sys.exit(1)

    video_stem = video_path.stem
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("detector")
    logger.info("视频: %s", video_path)
    logger.info("商品名: %s", product_name)
    logger.info("输出目录: %s", output_dir)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("无法打开视频文件")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    step_frames = max(1, int(fps * sample_interval))

    logger.info("FPS=%.2f  总帧数=%d  时长=%.1fs  抽帧间隔=%d帧", fps, total_frames, duration_sec, step_frames)

    hits = []
    all_results = []
    frame_idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps
        # 文件名：视频名_时间码.jpg（时间码冒号替换为-避免非法字符）
        tc_str = seconds_to_tc(time_sec).replace(":", "-")
        frame_filename = f"{video_stem}_{tc_str}.jpg"
        frame_path = frames_dir / frame_filename

        # 保存帧
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        logger.info("[%.1fs] 分析帧: %s", time_sec, frame_filename)
        result = query_moondream(str(frame_path), product_name, logger)
        score = result["score"]
        logger.info("  → score=%d  category=%s", score, result["category"])

        frame_record = {
            "time": round(time_sec, 3),
            "tc": seconds_to_tc(time_sec),
            "frame_file": frame_filename,
            "score": score,
            "category": result["category"],
            "reason": result["reason"]
        }
        all_results.append(frame_record)

        if score >= threshold:
            hits.append({"time": time_sec, "score": score, "category": result["category"]})

        # 如果不保留帧，分析完就删
        if not keep_frames:
            try:
                os.remove(frame_path)
            except Exception:
                pass

        frame_idx += step_frames

    cap.release()
    logger.info("抽帧分析完成，共 %d 帧，命中 %d 帧", len(all_results), len(hits))

    # 保存完整 result.json
    full_result = {
        "video": video_stem,
        "video_path": str(video_path),
        "product_name": product_name,
        "run_time": run_time,
        "total_frames_sampled": len(all_results),
        "hit_frames": len(hits),
        "frames": all_results
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)
    logger.info("result.json 已保存")

    # 合并窗口
    windows = merge_hits_to_windows(sorted(hits, key=lambda x: x["time"]))
    logger.info("合并后窗口数: %d", len(windows))
    for w in windows:
        logger.info("  [%s → %s]  avg_score=%.1f  %s", w["start_tc"], w["end_tc"], w["avg_score"], w["dominant_category"])

    # 保存 result_simple.json
    simple = {
        "video": video_stem,
        "suggested_windows": windows
    }
    simple_path = output_dir / "result_simple.json"
    with open(simple_path, "w", encoding="utf-8") as f:
        json.dump(simple, f, ensure_ascii=False, indent=2)
    logger.info("result_simple.json 已保存: %s", simple_path)

    # 控制台汇总
    print("\n" + "="*60)
    print(f"✅ 分析完成  商品: {product_name}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 命中窗口数: {len(windows)}")
    for w in windows:
        print(f"   {w['start_tc']} → {w['end_tc']}  ({w['duration']}s)  avg={w['avg_score']}  [{w['dominant_category']}]")
    print("="*60)

    return str(simple_path)


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="使用 Ollama moondream 定位视频中商品出现的时间窗口"
    )
    parser.add_argument("video_path", help="源视频完整路径")
    parser.add_argument("product_name", help='商品名，例如："白色帽子"')
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        default=True,
        help="是否保留抽帧 jpg（默认保留）"
    )
    parser.add_argument(
        "--no-keep-frames",
        action="store_true",
        default=False,
        help="分析后删除抽帧 jpg"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=SAMPLE_INTERVAL_SEC,
        help=f"抽帧间隔秒数（默认 {SAMPLE_INTERVAL_SEC}）"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=CONFIDENCE_THRESHOLD,
        help=f"命中分数阈值（默认 {CONFIDENCE_THRESHOLD}）"
    )
    args = parser.parse_args()

    keep = args.keep_frames and not args.no_keep_frames
    process_video(args.video_path, args.product_name, keep, args.interval, args.threshold)


if __name__ == "__main__":
    main()