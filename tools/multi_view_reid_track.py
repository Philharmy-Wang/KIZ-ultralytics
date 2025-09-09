#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多视角 ReID 跟踪（官方 ReID + 进度条 + 实时流支持 + 全流程日志）
================================================================
满足需求：
1) 持久化跟踪（persist=True）
2) 随时间绘制轨迹（自定义折线）
3) 多线程跟踪（每路视频独立线程与 YOLO 实例）
4) 启用 ReID（Ultralytics 官方默认：with_reid=True, model=auto）
5) 输出处理/运行过程：控制台与文件日志、每路 summary.json、全局 aggregate.csv
6) 新增：文件源显示 tqdm 进度条；实时流（摄像头/RTSP/HTTP）进入无进度条实时模式并周期性输出统计日志

编程思路（嵌入式说明）：
- 用 logging 统一输出（含线程名），既在控制台打印，也写入文件。
- 每路视频维护运行统计：累计帧数、唯一 ID 集合、平均 FPS 等；对“文件源”显示 tqdm 进度条，对“实时流源”按 LOG_INTERVAL 帧打印统计。
- 以 stream=True 逐帧获取结果，自绘框/ID/轨迹，自己写视频，避免 CLI 参数差异。
- ReID：对 botsort.yaml 打补丁（with_reid=True, model=auto），交给 Ultralytics 自动选择/下载官方 ReID 模型。
- 实时流常见问题：首帧未到时无法知道尺寸 → 使用“惰性创建 VideoWriter”（拿到首帧再创建）。
"""

from __future__ import annotations
import os, cv2, yaml, json, csv, time, hashlib, logging, threading
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional

import torch
from ultralytics import YOLO

# 进度条：若环境无 tqdm 则优雅降级为仅日志
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from itertools import count
_PROGRESS_POS = count(0)   # 多线程下给每路视频分配不同的 tqdm 行位

# =========================
# ① 可配置参数（根据你的环境修改）
# =========================

# 检测模型（你的 best.pt）
MODEL_PATH = "/mnt/data/wgb/ultralytics/runs/detect/train/weights/best.pt"

# 你的 botsort.yaml（Ultralytics 自带）
TRACKER_BASE_YAML = "/mnt/data/wgb/ultralytics/ultralytics/cfg/trackers/botsort.yaml"

# 多路输入（本地文件 / 摄像头ID / rtsp/http 链接 都支持）：
#   - 文件会显示进度条；
#   - 流（摄像头/rtsp/http）为实时模式，不显示进度条，仅周期性日志。
# VIDEOS = [
#     # 文件（你当前加速后的 12× 视频）
#     "/mnt/data/wgb/ultralytics/demo_video/rec-16328-con-20250410142418-camera-0_x12.mp4",
#     "/mnt/data/wgb/ultralytics/demo_video/rec-16328-con-20250410142418-camera-1_x12.mp4",
#     "/mnt/data/wgb/ultralytics/demo_video/rec-16328-con-20250410142418-camera-2_x12.mp4",
#     "/mnt/data/wgb/ultralytics/demo_video/rec-16328-con-20250410142418-camera-3_x12.mp4",
#     # 例子：本机摄像头
#     # "0",
#     # 例子：RTSP 网络摄像头
#     # "rtsp://user:pass@192.168.1.10:554/stream1",
#     # 例子：HTTP 在线视频
#     # "https://example.com/some_video.mp4",
# ]

VIDEOS = [
    #  "https://www.bilibili.com/video/BV1UvcRehE8g/?share_source=copy_web&vd_source=80dfdf4ec93c91a52b688cd7bd44a8c9"
    # 文件（你当前加速后的 12× 视频）
    "/mnt/data/wgb/ultralytics/demo_video/rec-16328-con-20250410142418-camera-0_x12.mp4"
    # 例子：本机摄像头
    # "0",
    # 例子：RTSP 网络摄像头
    # "rtsp://user:pass@192.168.1.10:554/stream1",
    # 例子：HTTP 在线视频
    # "https://example.com/some_video.mp4",
]

# GPU 轮询分配（按视频索引循环使用）
GPU_DEVICES = ["0", "1"]   # 只有一张卡就 ["0"]

# 最大并发线程数（通常与视频数一致；若显存紧张可调小）
MAX_THREADS = 4

# 输出根目录
OUTPUT_DIR = "/mnt/data/wgb/ultralytics/runs/track/multi_view_reid_official_logs"

# 推理参数
IMGSZ = 640
CONF  = 0.25
IOU   = 0.45
HALF  = True               # 支持 FP16 的显卡建议 True
VID_STRIDE = 1             # 2/3 可跳帧提速
SAVE_FPS: Optional[int] = None  # None=沿用源视频 FPS（流拿不到 FPS 时用 25）

# 可视化参数
LINE_THICK = 2
DRAW_ID_TEXT = True
MAX_TRAIL = 600   # 每个目标保留多少个历史点
TRAJ_STEP = 1    # 每 TRAJ_STEP 帧记录一次轨迹点

# 记录运行过程的粒度
LOG_INTERVAL = 50  # 每处理多少帧打印一次进度/统计
SAVE_TXT = True    # 另存 MOT 简化文本（便于后处理/评估）


# =========================
# ② 日志系统
# =========================

def setup_logging(run_root: Path) -> logging.Logger:
    """
    配置全局日志：
    - 控制台 Handler（INFO）
    - 文件 Handler（DEBUG），保存到 run_root/run.log
    """
    run_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multi_view_reid")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(threadName)s] %(message)s", "%H:%M:%S")

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件
    fh = logging.FileHandler(run_root / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# =========================
# ③ ReID 补丁：官方默认
# =========================

def ensure_official_reid_yaml(base_yaml: str, out_dir: str, logger: logging.Logger) -> str:
    """
    读取 botsort.yaml，强制：
      - with_reid: True
      - model: auto  （Ultralytics 自动选择/下载官方 ReID 分类模型）
    其余关键阈值若缺失则补默认。
    """
    base = Path(base_yaml)
    assert base.exists(), f"botsort.yaml 不存在：{base_yaml}"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

    cfg.setdefault("tracker_type", "botsort")
    cfg["with_reid"] = True
    cfg["model"] = "auto"

    # 追踪阈值（若原文件没配则补）
    cfg.setdefault("track_high_thresh", 0.5)
    cfg.setdefault("track_low_thresh", 0.1)
    cfg.setdefault("new_track_thresh", 0.6)
    cfg.setdefault("match_thresh", 0.8)
    cfg.setdefault("track_buffer", 30)
    cfg.setdefault("visualize", False)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    patched = out_dir / "botsort_official_reid.yaml"
    patched.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    logger.info(f"ReID 补丁完成：with_reid=True, model=auto -> {patched}")
    return str(patched.resolve())


# =========================
# ④ 工具：源解析 / 元信息探测 / 可视化 / MOT
# =========================

def is_file_path(s: str) -> bool:
    """判断 source 是否为本地文件路径（存在即算文件）"""
    try:
        p = Path(s)
        return p.exists() and p.is_file()
    except Exception:
        return False

def parse_source_and_name(src: str) -> Tuple[object, str, bool]:
    """
    解析 source：
      - 本地文件：返回 (绝对路径字符串, '文件名', is_stream=False)
      - 摄像头编号 '0'/'1'：返回 (int(id), 'cam{id}', is_stream=True)
      - URL (rtsp/rtmp/http/https)：返回 (原字符串, '流的简名', is_stream=True)
    """
    s = str(src).strip()
    if s.isdigit():  # 摄像头 ID
        return int(s), f"cam{s}", True

    low = s.lower()
    if low.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        # 简单可读名（去协议/认证信息/查询参数）
        base = s.split("://", 1)[-1]
        base = base.split("?")[0].split("@")[-1].replace("/", "_")
        base = base[:60] if len(base) > 60 else base
        name = f"stream_{base or 'live'}"
        return s, name, True

    if is_file_path(s):
        p = Path(s).resolve()
        return str(p), p.stem, False

    # 兜底：当成流
    return s, f"stream_{hashlib.md5(s.encode()).hexdigest()[:8]}", True

def probe_video_meta_with_cv2(src_for_cv2) -> Tuple[int, int, float, int]:
    """
    用 OpenCV 读取一遍元信息（宽高/FPS/总帧数）。
    - 对文件通常能拿到总帧数（>0）
    - 对实时流/部分容器，总帧数可能为 -1
    """
    cap = cv2.VideoCapture(src_for_cv2)
    if not cap.isOpened():
        return 0, 0, 0.0, -1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = -1
    cap.release()
    return w, h, fps, total

def color_from_id(track_id: int) -> Tuple[int, int, int]:
    """为同一 ID 生成稳定 BGR 颜色（避免过暗）"""
    h = int(hashlib.sha256(str(track_id).encode()).hexdigest(), 16)
    r, g, b = (h % 255), ((h // 255) % 255), ((h // (255*255)) % 255)
    def lift(x): return int(50 + (x * 205) / 255)
    return (lift(b), lift(g), lift(r))

def draw_boxes_ids_trails(frame, xyxy, ids, trails: Dict[int, deque], frame_idx: int):
    """在帧上画框、ID 与历史轨迹折线"""
    for box, tid in zip(xyxy, ids):
        x1, y1, x2, y2 = map(int, box.tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if tid not in trails:
            trails[tid] = deque(maxlen=MAX_TRAIL)
        if frame_idx % TRAJ_STEP == 0:
            trails[tid].append((cx, cy))

        color = color_from_id(int(tid))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICK)

        if DRAW_ID_TEXT:
            label = f"ID {int(tid)}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, LINE_THICK)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), LINE_THICK)

    for tid, pts in trails.items():
        if len(pts) < 2:
            continue
        color = color_from_id(int(tid))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, LINE_THICK)

def write_mot(txt: Path, frame_idx: int, ids, xyxy):
    """保存简化 MOT：frame,id,x,y,w,h,conf,-1,-1,-1（conf=1.0）"""
    with txt.open("a", encoding="utf-8") as f:
        for tid, box in zip(ids, xyxy):
            x1, y1, x2, y2 = map(float, box.tolist())
            w, h = x2 - x1, y2 - y1
            f.write(f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1\n")


# =========================
# ⑤ 运行统计与摘要
# =========================

class RunStats:
    """记录每路视频的运行统计，用于周期性日志与最终 summary"""
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()
        self.frames = 0
        self.total_dets = 0
        self.unique_ids = set()
        self.last_log_t = self.start
        self.max_concurrent = 0  # 单帧最多目标数

    def step(self, det_ids: List[int]):
        self.frames += 1
        n = len(det_ids)
        self.total_dets += n
        self.max_concurrent = max(self.max_concurrent, n)
        for i in det_ids:
            self.unique_ids.add(int(i))

    def snapshot(self) -> Dict:
        now = time.time()
        dur = max(1e-6, now - self.start)
        fps = self.frames / dur
        return dict(
            name=self.name,
            frames=self.frames,
            duration_sec=dur,
            avg_fps=fps,
            total_dets=self.total_dets,
            unique_ids=len(self.unique_ids),
            max_concurrent=self.max_concurrent
        )

    def dump_json(self, path: Path):
        path.write_text(json.dumps(self.snapshot(), indent=2, ensure_ascii=False), encoding="utf-8")


# =========================
# ⑥ 单路视频/流：持久化 + 进度条/日志
# =========================

def track_one_video(video_path: str, device: str, tracker_yaml: str, out_root: Path, glog: logging.Logger):
    """
    单路视频/流 跟踪（逐帧、自绘、持久化）：
    - 若为本地文件：显示 tqdm 进度条（百分比/ETA）
    - 若为实时流：无进度条，仅周期性日志；输出视频延迟创建（拿到首帧尺寸后再建）
    """
    # 解析源（得到：可被 OpenCV/YOLO 接受的 source、友好名字、是否流）
    src_parsed, nice_name, is_stream = parse_source_and_name(video_path)

    # 每路独立日志文件（贴近该名字）
    logger = logging.getLogger(f"video-{nice_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(threadName)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_root / f"{nice_name}_run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 元信息（对文件通常能拿到总帧数；对流多为 -1）
    src_w, src_h, src_fps, total_frames = probe_video_meta_with_cv2(src_parsed)
    if not is_stream and not is_file_path(str(video_path)):
        raise AssertionError(f"视频不存在：{video_path}")

    # 输出路径
    out_dir = out_root / nice_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{nice_name}_tracked.mp4"
    out_txt  = out_dir / f"{nice_name}_tracks.txt"
    summary_json = out_dir / f"{nice_name}_summary.json"

    # 输出 FPS：优先 SAVE_FPS，否则用源 FPS（拿不到则用 25）
    save_fps = SAVE_FPS or (src_fps if src_fps and src_fps > 0 else 25.0)

    # 惰性创建 VideoWriter（首帧到手后再创建，避免未知尺寸报错）
    vw = None
    def ensure_writer(w: int, h: int):
        nonlocal vw
        if vw is None:
            vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                 save_fps, (w, h), True)

    # 加载模型
    model = YOLO(MODEL_PATH)
    use_half = HALF and torch.cuda.is_available()
    logger.info(
        f"开始处理：{nice_name} | device={device} | tracker={tracker_yaml} | "
        f"is_stream={is_stream} | src_fps={src_fps} | total={total_frames}"
    )
    logger.info(f"模型加载完成（half={use_half}, imgsz={IMGSZ}, conf={CONF}, iou={IOU}, vid_stride={VID_STRIDE}）")

    trails: Dict[int, deque] = {}
    stats = RunStats(nice_name)

    # 文件：开启进度条；流：无进度条
    pbar = None
    if (not is_stream) and total_frames > 0 and tqdm is not None:
        pbar = tqdm(total=total_frames, position=next(_PROGRESS_POS),
                    desc=nice_name, leave=True, ncols=100)

    # 流式逐帧 + 持久化
    gen = model.track(
        source=src_parsed,          # 文件/摄像头ID/URL 都可
        device=device,
        tracker=tracker_yaml,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        half=use_half,
        vid_stride=VID_STRIDE,
        persist=True,
        stream=True,               # 关键：逐帧拿结果
        save=False,
        verbose=False,
        show=False,                # 若本地有 GUI 可改 True 弹窗（远程一般不开）
    )

    try:
        for results in gen:
            # 注意：results.orig_img 已是 BGR uint8；不拷贝可省内存
            frame = results.orig_img
            h, w = frame.shape[:2]
            ensure_writer(w, h)

            boxes = getattr(results, "boxes", None)
            det_ids, xyxy = [], None
            if boxes is not None and boxes.id is not None and len(boxes) > 0:
                det_ids = boxes.id.int().tolist()
                xyxy = boxes.xyxy.cpu()
                draw_boxes_ids_trails(frame, xyxy, det_ids, trails, stats.frames)
                if SAVE_TXT:
                    write_mot(out_txt, stats.frames + 1, det_ids, xyxy)

            vw.write(frame)
            stats.step(det_ids)

            # 进度条（文件）或心跳日志（流）
            if pbar is not None:
                pbar.update(1)
            elif stats.frames % LOG_INTERVAL == 0:
                snap = stats.snapshot()
                logger.info(
                    f"{nice_name} [{stats.frames}] "
                    f"avgFPS={snap['avg_fps']:.2f} det={len(det_ids)} "
                    f"uniqID={snap['unique_ids']} maxConc={snap['max_concurrent']}"
                )

    except Exception as e:
        logger.exception(f"处理 {nice_name} 时出错：{e}")
        raise
    finally:
        if pbar is not None:
            pbar.close()
        if vw is not None:
            vw.release()

    # 结束与总结
    summary = stats.snapshot()
    summary.update(dict(
        video=str(video_path),
        is_stream=is_stream,
        output_video=str(out_mp4),
        mot_txt=str(out_txt if SAVE_TXT else ""),
        src_fps=src_fps,
        total_frames=total_frames
    ))
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    glog.info(
        f"[DONE] {nice_name} frames={summary['frames']} "
        f"avgFPS={summary['avg_fps']:.2f} uniqID={summary['unique_ids']} "
        f"out={out_mp4}"
    )


# =========================
# ⑦ 主入口：多线程 + 全局汇总
# =========================

def log_env(logger: logging.Logger):
    """打印环境信息（PyTorch/CUDA/GPU）"""
    logger.info(f"Torch: {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        ng = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(ng)]
        logger.info(f"GPU Count: {ng}  -> {names}")

def main():
    out_root = Path(OUTPUT_DIR).resolve()
    logger = setup_logging(out_root)
    log_env(logger)

    assert Path(MODEL_PATH).exists(), f"模型不存在：{MODEL_PATH}"
    assert Path(TRACKER_BASE_YAML).exists(), f"botsort.yaml 不存在：{TRACKER_BASE_YAML}"

    # ReID 补丁
    tracker_yaml = ensure_official_reid_yaml(TRACKER_BASE_YAML, str(out_root), logger)

    # 设备轮询分配
    devs = GPU_DEVICES if GPU_DEVICES else ["0"]
    max_threads = min(MAX_THREADS, len(VIDEOS))

    # 多线程批处理（简单队列式：一批跑完再下一批，避免挤爆显存）
    idx = 0
    all_summaries: List[Dict] = []
    while idx < len(VIDEOS):
        batch = []
        for _ in range(max_threads):
            if idx >= len(VIDEOS): break
            vid = VIDEOS[idx]
            dev = devs[idx % len(devs)]
            t = threading.Thread(
                target=track_one_video,
                kwargs=dict(video_path=vid, device=dev, tracker_yaml=tracker_yaml, out_root=out_root, glog=logger),
                name=f"cam{idx}-GPU{dev}",
                daemon=True
            )
            t.start()
            batch.append((idx, vid, t))
            logger.info(f"启动线程：cam{idx} -> {Path(str(vid)).name} on GPU{dev}")
            idx += 1

        # 等这一批完成
        for (i, vid, t) in batch:
            t.join()
            # 每路的 summary.json 在对应子目录里，汇总时再读
            # 注意：对流源 nice_name 与 stem 不同，这里用 parse_source_and_name 获取名字
            _, stem, _ = parse_source_and_name(vid)
            summary_json = out_root / stem / f"{stem}_summary.json"
            if summary_json.exists():
                try:
                    all_summaries.append(json.loads(summary_json.read_text(encoding="utf-8")))
                except Exception:
                    logger.warning(f"读取 summary 失败：{summary_json}")

    # 写一个全局 CSV 汇总
    agg_csv = out_root / "aggregate.csv"
    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "video", "frames", "duration_sec", "avg_fps",
            "total_dets", "unique_ids", "max_concurrent", "output_video", "mot_txt"
        ])
        writer.writeheader()
        for s in all_summaries:
            writer.writerow({
                "name": s.get("name", s.get("video", "")),
                "video": s.get("video", ""),
                "frames": s.get("frames", 0),
                "duration_sec": f"{s.get('duration_sec', 0):.2f}",
                "avg_fps": f"{s.get('avg_fps', 0):.2f}",
                "total_dets": s.get("total_dets", 0),
                "unique_ids": s.get("unique_ids", 0),
                "max_concurrent": s.get("max_concurrent", 0),
                "output_video": s.get("output_video", ""),
                "mot_txt": s.get("mot_txt", ""),
            })

    logger.info(f"[ALL DONE] 输出目录：{out_root} | 汇总：{agg_csv}")

if __name__ == "__main__":
    main()
