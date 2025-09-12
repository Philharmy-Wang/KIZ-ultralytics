#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多视角 ReID 跟踪（按停留时长的热力图 + 对比度增强 + 双路写出）
================================================================
改进点：
- 热力图强度 = 停留时长（秒），而非帧计数。
- 高斯核盖章到下采样网格，支持固定或随 bbox 自适应半径。
- 归一化/对比度：固定上限（clip 秒）或 p99，自带 gamma 提升低强度区域可见度。
- 维持：ReID、进度条、实时流、单目标稳定ID、MOT/日志/汇总、双路写出。

参考：Ultralytics 热图指南与模块文档（思路一致，代码自实现更可控）。
"""

from __future__ import annotations
import os, cv2, yaml, json, csv, time, hashlib, logging, threading
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from ultralytics import YOLO

# tqdm（无则降级为日志）
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
from itertools import count
_PROGRESS_POS = count(0)

# =========================
# ① 配置
# =========================

MODEL_PATH = "/mnt/data/wgb/ultralytics/runs/detect/train/weights/best.pt"
TRACKER_BASE_YAML = "/mnt/data/wgb/ultralytics/ultralytics/cfg/trackers/botsort.yaml"

VIDEOS = [
    "/mnt/data/wgb/ultralytics/demo_video/rec-16328-con-20250410142418-camera-0_x12.mp4",
    # 可混合流源："0" / "rtsp://user:pass@ip/stream" / "https://...mp4"
]

GPU_DEVICES = ["0", "1"]
MAX_THREADS = 4
OUTPUT_DIR = "/mnt/data/wgb/ultralytics/runs/track/multi_view_reid_official_logs_heatmap_heat"

IMGSZ = 640
CONF  = 0.25
IOU   = 0.45
HALF  = True
VID_STRIDE = 1
SAVE_FPS: Optional[int] = None

LINE_THICK = 2
DRAW_ID_TEXT = True
MAX_TRAIL = 0      # 0/负数 = 无限保留所有轨迹点；>0 只保留最近 N 个
TRAJ_STEP = 1

LOG_INTERVAL = 50
SAVE_TXT = True

# —— 热力图（按停留时长/秒） —— #
ENABLE_HEATMAP = True
HEATMAP_DOWNSCALE = 2          # 下采样倍数（越大越快，越平滑）
HEATMODE = "bbox"              # "fixed" 固定半径 | "bbox" 随 bbox 自适应
HEAT_RADIUS_PX = 8             # HEATMODE=fixed 时使用的像素半径（下采样前坐标系）
HEAT_BBOX_SCALE = 0.25         # HEATMODE=bbox 时：radius ≈ scale * sqrt(w*h)
HEATMAP_ALPHA = 0.35           # 叠加透明度
HEATMAP_CMAP = cv2.COLORMAP_JET
HEATMAP_GAMMA = 0.6            # <1 提升暗部；1 等比；>1 压暗暗部
CLIP_MAX_SECONDS: Optional[float] = 60.0  # 例：60.0 → 停留60s达满红；None 用 p99
STREAM_FPS_ASSUME = 25.0       # 流源拿不到 FPS 时的估计
DT_MIN, DT_MAX = 0.001, 0.2    # 流源 dt 的夹持（秒）

# 双路写出：原始可视化 + 热力图叠加
WRITE_BOTH_VIDEOS = True

# —— 单目标ID稳定（你的视频常常只有1只猴） —— #
SINGLE_TARGET = True
STABLE_ID = 1
PRIMARY_IOU_BIAS = 0.1
MIN_IOU_KEEP = 0.1

# —— 可选类别筛选（只对这些类计入热图；None 表示不过滤） —— #
TARGET_CLASS_IDS: Optional[List[int]] = None  # 如只算猴子： [0]

# =========================
# ② 日志
# =========================
def setup_logging(run_root: Path) -> logging.Logger:
    run_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multi_view_reid")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(threadName)s] %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(run_root / "run.log", encoding="utf-8"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger

# =========================
# ③ ReID 补丁（更保守阈值）
# =========================
def ensure_official_reid_yaml(base_yaml: str, out_dir: str, logger: logging.Logger) -> str:
    base = Path(base_yaml)
    assert base.exists(), f"botsort.yaml 不存在：{base_yaml}"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

    cfg.setdefault("tracker_type", "botsort")
    cfg["with_reid"] = True
    cfg["model"] = "auto"

    cfg["track_high_thresh"] = float(cfg.get("track_high_thresh", 0.5))
    cfg["track_low_thresh"]  = float(cfg.get("track_low_thresh", 0.1))
    cfg["new_track_thresh"]  = max(0.6, float(cfg.get("new_track_thresh", 0.6)))
    cfg["match_thresh"]      = max(0.9, float(cfg.get("match_thresh", 0.8)))
    cfg["track_buffer"]      = max(60, int(cfg.get("track_buffer", 30)))
    cfg.setdefault("visualize", False)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    patched = out_dir / "botsort_official_reid.yaml"
    patched.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info(f"ReID 补丁：with_reid=True, match_thresh={cfg['match_thresh']}, track_buffer={cfg['track_buffer']} -> {patched}")
    return str(patched.resolve())

# =========================
# ④ 工具：源解析 / 元信息 / 绘制 / MOT
# =========================
def is_file_path(s: str) -> bool:
    try:
        p = Path(s); return p.exists() and p.is_file()
    except Exception:
        return False

def parse_source_and_name(src: str) -> Tuple[object, str, bool]:
    s = str(src).strip()
    if s.isdigit():  # 摄像头
        return int(s), f"cam{s}", True
    low = s.lower()
    if low.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        base = s.split("://", 1)[-1]; base = base.split("?")[0].split("@")[-1].replace("/", "_")
        base = base[:60] if len(base) > 60 else base
        return s, f"stream_{base or 'live'}", True
    if is_file_path(s):
        p = Path(s).resolve(); return str(p), p.stem, False
    return s, f"stream_{hashlib.md5(s.encode()).hexdigest()[:8]}", True

def probe_video_meta_with_cv2(src_for_cv2) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(src_for_cv2)
    if not cap.isOpened(): return 0, 0, 0.0, -1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); total = total if total > 0 else -1
    cap.release(); return w, h, fps, total

def color_from_id(track_id: int) -> Tuple[int, int, int]:
    h = int(hashlib.sha256(str(track_id).encode()).hexdigest(), 16)
    r, g, b = (h % 255), ((h // 255) % 255), ((h // (255*255)) % 255)
    def lift(x): return int(50 + (x * 205) / 255)
    return (lift(b), lift(g), lift(r))

def append_trail(trails: Dict[int, object], tid: int, pt: Tuple[int,int]):
    if tid not in trails:
        if MAX_TRAIL and MAX_TRAIL > 0:
            trails[tid] = deque(maxlen=MAX_TRAIL)
        else:
            trails[tid] = []  # 无限
    trails[tid].append(pt)

def draw_boxes_ids_trails(frame, xyxy, ids, trails: Dict[int, object], frame_idx: int):
    for box, tid in zip(xyxy, ids):
        x1, y1, x2, y2 = map(int, box.tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if frame_idx % TRAJ_STEP == 0:
            append_trail(trails, int(tid), (cx, cy))
        color = color_from_id(int(tid))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICK)
        if DRAW_ID_TEXT:
            label = f"ID {int(tid)}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, LINE_THICK)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), LINE_THICK)
    # 轨迹
    for tid, pts in trails.items():
        if len(pts) < 2: continue
        color = color_from_id(int(tid))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], color, LINE_THICK)

def write_mot(txt: Path, frame_idx: int, ids, xyxy):
    with txt.open("a", encoding="utf-8") as f:
        for tid, box in zip(ids, xyxy):
            x1, y1, x2, y2 = map(float, box.tolist())
            w, h = x2 - x1, y2 - y1
            f.write(f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1\n")

# =========================
# ⑤ 热力图（停留时长秒）实现
# =========================
class DwellHeatmap:
    """
    将目标中心点按 dt(秒) 权重，用高斯核累加到下采样网格；累计矩阵的单位是“秒”。
    """
    def __init__(self, downscale:int, mode:str, radius_px:int, bbox_scale:float):
        self.accum: Optional[np.ndarray] = None
        self.ds = max(1, int(downscale))
        self.mode = mode
        self.radius_px = max(1, int(radius_px))
        self.bbox_scale = float(bbox_scale)
        self.kernels = {}  # 半径->预计算高斯核（归一化 sum=1）

    def _ensure(self, w:int, h:int):
        if self.accum is None:
            aw, ah = max(1, w // self.ds), max(1, h // self.ds)
            self.accum = np.zeros((ah, aw), dtype=np.float32)

    @staticmethod
    def _gaussian2d(ks:int, sigma:float) -> np.ndarray:
        ax = np.arange(-ks//2 + 1., ks//2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        k = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        s = k.sum(); 
        return (k / (s + 1e-8)).astype(np.float32)

    def _get_kernel(self, px_radius:int) -> np.ndarray:
        # 令 kernel 尺寸 ~ 6*sigma，sigma ~ px_radius/2
        px_radius = max(1, int(px_radius))
        sigma = max(1.0, px_radius / 2.0)
        ks = int(6 * sigma) | 1  # 奇数
        key = (ks, round(sigma,1))
        if key not in self.kernels:
            self.kernels[key] = self._gaussian2d(ks, sigma)
        return self.kernels[key]

    def _radius_for_box(self, x1:int, y1:int, x2:int, y2:int) -> int:
        if self.mode == "fixed":
            return self.radius_px
        # bbox 自适应：对角线近似 ~ sqrt(w*h)*sqrt(2)；简化用 sqrt(w*h)
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        est = int(self.bbox_scale * np.sqrt(w * h))
        return max(2, est)

    def add(self, boxes_xyxy: np.ndarray, dt_sec: float):
        """
        在每个目标中心处“盖章”一个高斯核，核强度乘以 dt_sec，使得累计量以秒为单位。
        """
        if self.accum is None:
            raise RuntimeError("heatmap not initialized yet")

        H, W = self.accum.shape
        for b in boxes_xyxy:
            x1, y1, x2, y2 = map(int, b.tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # 下采样坐标
            xx, yy = cx // self.ds, cy // self.ds

            r = self._radius_for_box(x1, y1, x2, y2) // self.ds
            r = max(2, r)
            ker = self._get_kernel(r)
            kh, kw = ker.shape
            # 将 kernel 覆盖到以 (yy,xx) 为中心的 ROI
            y0 = yy - kh//2; y1_ = y0 + kh
            x0 = xx - kw//2; x1_ = x0 + kw

            # 与边界相交
            ry0, ry1 = max(0, y0), min(H, y1_)
            rx0, rx1 = max(0, x0), min(W, x1_)
            ky0, ky1 = ry0 - y0, ry1 - y0
            kx0, kx1 = rx0 - x0, rx1 - x0
            if ry1 <= ry0 or rx1 <= rx0: 
                continue

            # 盖章（乘以 dt_sec）
            self.accum[ry0:ry1, rx0:rx1] += (ker[ky0:ky1, kx0:kx1] * dt_sec).astype(np.float32)

    def render(self, out_size: Tuple[int,int], clip_seconds: Optional[float], gamma: float, cmap:int) -> np.ndarray:
        acc = self.accum
        if acc is None:
            return None
        if clip_seconds and clip_seconds > 0:
            vmax = clip_seconds
        else:
            vmax = max(1e-3, np.percentile(acc, 99))  # p99 自适应
        norm = np.clip(acc / (vmax + 1e-6), 0.0, 1.0)
        if gamma and gamma > 0:
            norm = np.power(norm, 1.0 / float(gamma))
        u8 = (norm * 255.0 + 0.5).astype(np.uint8)
        up = cv2.resize(u8, out_size, interpolation=cv2.INTER_CUBIC)
        color = cv2.applyColorMap(up, cmap)
        return color

def overlay_heatmap(frame: np.ndarray, heat_color: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(heat_color, alpha, frame, 1.0 - alpha, 0.0)

# =========================
# ⑥ 统计
# =========================
class RunStats:
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()
        self.frames = 0
        self.total_dets = 0
        self.unique_ids = set()
        self.max_concurrent = 0
    def step(self, det_ids: List[int]):
        self.frames += 1
        n = len(det_ids)
        self.total_dets += n
        self.max_concurrent = max(self.max_concurrent, n)
        for i in det_ids: self.unique_ids.add(int(i))
    def snapshot(self) -> Dict:
        dur = max(1e-6, time.time() - self.start)
        return dict(
            name=self.name, frames=self.frames, duration_sec=dur,
            avg_fps=self.frames/dur, total_dets=self.total_dets,
            unique_ids=len(self.unique_ids), max_concurrent=self.max_concurrent
        )

# =========================
# ⑦ 单目标选择与ID稳定
# =========================
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, xx2-xx1), max(0.0, yy2-yy1)
    inter = iw*ih
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[2]-b[0])*(b[3]-b[1])
    union = max(area_a+area_b-inter, 1e-6)
    return float(inter/union)

def select_primary_box(prev_box: Optional[np.ndarray], boxes_xyxy: np.ndarray, conf: np.ndarray) -> int:
    if boxes_xyxy.shape[0] == 1:
        return 0
    if prev_box is not None:
        ious = np.array([iou_xyxy(prev_box, b) for b in boxes_xyxy], dtype=np.float32)
        best_i = int(np.argmax(ious))
        if ious[best_i] >= MIN_IOU_KEEP + 1e-6:
            scores = ious + PRIMARY_IOU_BIAS * (conf / (conf.max()+1e-6))
            return int(np.argmax(scores))
    return int(np.argmax(conf))

# =========================
# ⑧ 单路处理（双路写出 + 停留时长热力图）
# =========================
def track_one_video(video_path: str, device: str, tracker_yaml: str, out_root: Path, glog: logging.Logger):
    # 解析源
    src_parsed, nice_name, is_stream = parse_source_and_name(video_path)

    # 每路日志
    logger = logging.getLogger(f"video-{nice_name}")
    logger.setLevel(logging.DEBUG); logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(threadName)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_root / f"{nice_name}_run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); logger.addHandler(fh)

    # 元信息
    src_w, src_h, src_fps, total_frames = probe_video_meta_with_cv2(src_parsed)
    if not is_stream and not is_file_path(str(video_path)):
        raise AssertionError(f"视频不存在：{video_path}")

    out_dir = out_root / nice_name; out_dir.mkdir(parents=True, exist_ok=True)
    out_plain_path   = out_dir / f"{nice_name}_tracked.mp4"
    out_overlay_path = out_dir / f"{nice_name}_tracked_heat.mp4"
    out_txt  = out_dir / f"{nice_name}_tracks.txt"
    summary_json = out_dir / f"{nice_name}_summary.json"

    save_fps = SAVE_FPS or (src_fps if src_fps and src_fps > 0 else 25.0)

    # 惰性写出
    vw_plain = vw_overlay = None
    def ensure_writers(w: int, h: int):
        nonlocal vw_plain, vw_overlay
        if vw_plain is None:
            vw_plain = cv2.VideoWriter(str(out_plain_path), cv2.VideoWriter_fourcc(*"mp4v"), save_fps, (w, h), True)
        if WRITE_BOTH_VIDEOS and vw_overlay is None:
            vw_overlay = cv2.VideoWriter(str(out_overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), save_fps, (w, h), True)

    # 模型
    model = YOLO(MODEL_PATH)
    use_half = HALF and torch.cuda.is_available()
    logger.info(f"开始处理：{nice_name} | device={device} | is_stream={is_stream} | src_fps={src_fps} | total={total_frames}")
    logger.info(f"模型加载完成（half={use_half}, imgsz={IMGSZ}, conf={CONF}, iou={IOU}, vid_stride={VID_STRIDE}）")

    trails: Dict[int, object] = {}
    stats = RunStats(nice_name)

    # 热力图（按秒累计）
    hm = DwellHeatmap(HEATMAP_DOWNSCALE, HEATMODE, HEAT_RADIUS_PX, HEAT_BBOX_SCALE)

    # 进度条
    pbar = None
    if (not is_stream) and total_frames > 0 and tqdm is not None:
        pbar = tqdm(total=total_frames, position=next(_PROGRESS_POS), desc=nice_name, leave=True, ncols=100)

    prev_primary_box: Optional[np.ndarray] = None
    last_t = time.perf_counter()

    # 流式推理
    gen = model.track(
        source=src_parsed, device=device, tracker=tracker_yaml,
        imgsz=IMGSZ, conf=CONF, iou=IOU, half=use_half, vid_stride=VID_STRIDE,
        persist=True, stream=True, save=False, verbose=False, show=False
    )

    try:
        for results in gen:
            frame = results.orig_img
            h, w = frame.shape[:2]
            ensure_writers(w, h)
            hm._ensure(w, h)

            # —— 计算本帧 dt（秒）——
            if is_stream:
                now = time.perf_counter()
                dt = max(DT_MIN, min(DT_MAX, now - last_t))
                last_t = now
            else:
                fps = src_fps if (src_fps and src_fps > 0) else STREAM_FPS_ASSUME
                dt = 1.0 / float(fps)

            boxes = getattr(results, "boxes", None)
            det_ids, xyxy = [], None

            if boxes is not None and len(boxes) > 0:
                # 可选类别筛选（只对猴子等目标生效）
                keep_idx = np.arange(len(boxes))
                if TARGET_CLASS_IDS is not None and getattr(boxes, "cls", None) is not None:
                    cls_arr = boxes.cls.cpu().numpy().astype(int)
                    mask = np.isin(cls_arr, np.array(TARGET_CLASS_IDS, dtype=int))
                    keep_idx = np.where(mask)[0]
                    if keep_idx.size == 0:
                        xyxy = None
                    else:
                        boxes = boxes[keep_idx]

                if boxes is not None and len(boxes) > 0:
                    xyxy_all = boxes.xyxy.cpu().numpy()
                    conf_all = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones((len(boxes),), np.float32)

                    if SINGLE_TARGET:
                        idx = select_primary_box(prev_primary_box, xyxy_all, conf_all)
                        xyxy = torch.from_numpy(xyxy_all[idx:idx+1])
                        det_ids = [STABLE_ID]
                        prev_primary_box = xyxy_all[idx]
                    else:
                        if boxes.id is not None:
                            det_ids = boxes.id.int().tolist()
                        else:
                            det_ids = list(range(1, len(boxes)+1))
                        xyxy = boxes.xyxy.cpu()
                else:
                    prev_primary_box = None
                    xyxy = None
            else:
                prev_primary_box = None

            # —— 可视化（plain）——
            plain_frame = frame.copy()
            if xyxy is not None and len(det_ids) > 0:
                draw_boxes_ids_trails(plain_frame, xyxy, det_ids, trails, stats.frames)
                if SAVE_TXT:
                    write_mot(out_txt, stats.frames + 1, det_ids, xyxy)

                # —— 按秒计入热力图 —— #
                # 使用 bbox 自适应/固定半径的高斯核盖章，核总和=1；乘以 dt -> 秒
                hm.add(xyxy.cpu().numpy(), float(dt))

            # —— 叠加帧 —— #
            if WRITE_BOTH_VIDEOS and ENABLE_HEATMAP:
                heat = hm.render((w, h), CLIP_MAX_SECONDS, HEATMAP_GAMMA, HEATMAP_CMAP)
                overlay_frame = overlay_heatmap(plain_frame, heat, HEATMAP_ALPHA) if heat is not None else plain_frame
            else:
                overlay_frame = None

            # 写帧
            if vw_plain is not None:
                vw_plain.write(plain_frame)
            if WRITE_BOTH_VIDEOS and vw_overlay is not None:
                vw_overlay.write(overlay_frame if overlay_frame is not None else plain_frame)

            stats.step(det_ids)

            # 进度条/心跳
            if pbar is not None:
                pbar.update(1)
            elif stats.frames % LOG_INTERVAL == 0:
                snap = stats.snapshot()
                logger.info(f"{nice_name} [{stats.frames}] avgFPS={snap['avg_fps']:.2f} det={len(det_ids)} uniqID={snap['unique_ids']} maxConc={snap['max_concurrent']}")

    except Exception as e:
        logger.exception(f"处理 {nice_name} 时出错：{e}")
        raise
    finally:
        if pbar is not None: pbar.close()
        if vw_plain is not None: vw_plain.release()
        if vw_overlay is not None: vw_overlay.release()

    # —— 输出热力图 PNG —— #
    heat = hm.render((w, h), CLIP_MAX_SECONDS, HEATMAP_GAMMA, HEATMAP_CMAP) if (ENABLE_HEATMAP and hm.accum is not None) else None
    if heat is not None:
        cv2.imwrite(str(out_dir / f"{nice_name}_heatmap.png"), heat)
        # 叠加版 PNG
        base = np.zeros_like(heat)  # 黑底
        overlay_img = overlay_heatmap(base, heat, HEATMAP_ALPHA)
        cv2.imwrite(str(out_dir / f"{nice_name}_heatmap_overlay.png"), overlay_img)

    # 总结
    summary = stats.snapshot()
    summary.update(dict(
        video=str(video_path), is_stream=is_stream,
        output_video_plain=str(out_plain_path),
        output_video_overlay=str(out_overlay_path) if WRITE_BOTH_VIDEOS else "",
        mot_txt=str(out_txt if SAVE_TXT else ""),
        src_fps=src_fps, total_frames=total_frames,
        heatmap=str(out_dir / f"{nice_name}_heatmap.png") if heat is not None else ""
    ))
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    glog.info(f"[DONE] {nice_name} frames={summary['frames']} avgFPS={summary['avg_fps']:.2f} "
              f"plain={out_plain_path} overlay={out_overlay_path if WRITE_BOTH_VIDEOS else 'N/A'}")

# =========================
# ⑨ 主入口：多线程 + 汇总
# =========================
def log_env(logger: logging.Logger):
    logger.info(f"Torch: {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        ng = torch.cuda.device_count(); names = [torch.cuda.get_device_name(i) for i in range(ng)]
        logger.info(f"GPU Count: {ng} -> {names}")

def main():
    out_root = Path(OUTPUT_DIR).resolve()
    logger = setup_logging(out_root)
    log_env(logger)

    assert Path(MODEL_PATH).exists(), f"模型不存在：{MODEL_PATH}"
    assert Path(TRACKER_BASE_YAML).exists(), f"botsort.yaml 不存在：{TRACKER_BASE_YAML}"

    tracker_yaml = ensure_official_reid_yaml(TRACKER_BASE_YAML, str(out_root), logger)

    devs = GPU_DEVICES if GPU_DEVICES else ["0"]
    max_threads = min(MAX_THREADS, len(VIDEOS))

    idx = 0
    all_summaries: List[Dict] = []
    while idx < len(VIDEOS):
        batch = []
        for _ in range(max_threads):
            if idx >= len(VIDEOS): break
            vid = VIDEOS[idx]; dev = devs[idx % len(devs)]
            t = threading.Thread(
                target=track_one_video,
                kwargs=dict(video_path=vid, device=dev, tracker_yaml=tracker_yaml, out_root=out_root, glog=logger),
                name=f"cam{idx}-GPU{dev}", daemon=True
            )
            t.start()
            batch.append((idx, vid, t))
            logger.info(f"启动线程：cam{idx} -> {Path(str(vid)).name} on GPU{dev}")
            idx += 1

        for (i, vid, t) in batch:
            t.join()
            _, stem, _ = parse_source_and_name(vid)
            summary_json = out_root / stem / f"{stem}_summary.json"
            if summary_json.exists():
                try:
                    all_summaries.append(json.loads(summary_json.read_text(encoding="utf-8")))
                except Exception:
                    logger.warning(f"读取 summary 失败：{summary_json}")

    agg_csv = out_root / "aggregate.csv"
    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "video", "frames", "duration_sec", "avg_fps",
            "total_dets", "unique_ids", "max_concurrent",
            "output_video_plain", "output_video_overlay", "mot_txt", "heatmap"
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
                "output_video_plain": s.get("output_video_plain", ""),
                "output_video_overlay": s.get("output_video_overlay", ""),
                "mot_txt": s.get("mot_txt", ""),
                "heatmap": s.get("heatmap", ""),
            })

    logger.info(f"[ALL DONE] 输出目录：{out_root} | 汇总：{agg_csv}")

if __name__ == "__main__":
    main()
