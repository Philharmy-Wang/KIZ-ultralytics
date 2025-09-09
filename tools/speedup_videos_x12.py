#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用 OpenCV 将一组视频加速 N 倍（真正减少帧数 → 推理也更快）
- 思路：读取源视频时“每 FACTOR 帧保留 1 帧”，写出时保持原 FPS。
- 输出：优先写 MP4（mp4v/avc1），不行则自动回退到 AVI（XVID/MJPG）。
- 不处理音频（跟踪/推理用不到，且 OpenCV 写音频不方便）。
- 不依赖系统 ffmpeg，适合无 sudo/网络不通环境。
"""

from __future__ import annotations
import cv2, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== 你的 4 个源视频 =====
VIDEOS = [
    "/mnt/data/wgb/ultralytics/demo_video/rec-16328-con-20250410142418-camera-0.avi",
    "/mnt/data/wgb/ultralytics/demo_video/rec-16328-con-20250410142418-camera-1.avi",
    "/mnt/data/wgb/ultralytics/demo_video/rec-16328-con-20250410142418-camera-2.avi",
    "/mnt/data/wgb/ultralytics/demo_video/rec-16328-con-20250410142418-camera-3.avi",
]

# ===== 只改这个参数就能调整倍速 =====
FACTOR = 12  # 加速倍数（整数 >= 2）

# ===== 写出编码器候选（按顺序尝试）=====
# 说明：不同系统/轮子支持不同编码器。我们依次尝试，直到创建成功为止。
CODEC_CANDIDATES = [
    ("mp4v", ".mp4"),  # MPEG-4 Part 2（通用）
    ("avc1", ".mp4"),  # H.264 FourCC（部分环境可用）
    ("XVID", ".avi"),  # Xvid（avi）
    ("MJPG", ".avi"),  # Motion JPEG（体积大但很稳）
]

LOG_EVERY = 200  # 每处理多少帧打印一次进度
MAX_PROCS = min(4, os.cpu_count() or 1)  # 并发进程数（与视频数取小）

def create_writer(base_out: Path, w: int, h: int, fps: float):
    """
    逐个尝试候选编码器，返回 (VideoWriter 对象, 实际输出路径)。
    - 若某编码器创建失败（isOpened=False），换下一个编码器/扩展名。
    """
    for fourcc_name, ext in CODEC_CANDIDATES:
        out_path = base_out.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), True)
        if vw.isOpened():
            return vw, out_path, fourcc_name
    return None, None, None

def speedup_one(src: str) -> tuple[str, bool, str]:
    """
    单个视频加速：
    - 每 FACTOR 帧保留 1 帧，写出时保持原 FPS（从而时长缩短为 1/FACTOR）。
    - 返回 (输出路径, 是否成功, 错误信息)。
    """
    t0 = time.time()
    pin = Path(src)
    if not pin.exists():
        return src, False, "源文件不存在"

    cap = cv2.VideoCapture(str(pin))
    if not cap.isOpened():
        return src, False, "无法打开视频（OpenCV 不支持该容器/编码或文件损坏）"

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    if FACTOR < 2 or int(FACTOR) != FACTOR:
        cap.release()
        return src, False, "FACTOR 应为整数且 >= 2"

    # 输出文件名：原名 + _x{FACTOR}（后缀由编码器决定）
    base_out = pin.with_name(pin.stem + f"_x{FACTOR}")

    vw, out_path, codec = create_writer(base_out, src_w, src_h, src_fps)
    if vw is None:
        cap.release()
        return src, False, "无法创建 VideoWriter（缺少合适编码器）"

    kept = 0
    readn = 0
    ok = True
    last_log = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 核心：每 FACTOR 帧保留 1 帧（readn 从 0 开始）
        if readn % FACTOR == 0:
            vw.write(frame)
            kept += 1
        readn += 1

        if readn % LOG_EVERY == 0:
            now = time.time()
            dt = max(1e-6, now - last_log)
            fps_rt = LOG_EVERY / dt
            last_log = now
            print(f"[{pin.name}] read={readn} kept={kept} rt_fps={fps_rt:.1f}")

    cap.release()
    vw.release()

    dur = time.time() - t0
    msg = f"OK codec={codec} out={out_path.name} kept={kept} / read={readn} time={dur:.2f}s"
    return str(out_path), ok, msg

def main():
    print(f"[INFO] 开始加速 {len(VIDEOS)} 个视频：FACTOR={FACTOR}（每 {FACTOR} 帧取 1 帧）")
    okn = badn = 0
    with ProcessPoolExecutor(max_workers=MAX_PROCS) as ex:
        futs = {ex.submit(speedup_one, v): v for v in VIDEOS}
        for fut in as_completed(futs):
            outp, ok, msg = fut.result()
            if ok:
                print(f"[OK ] {msg}")
                okn += 1
            else:
                print(f"[ERR] src={outp} -> {msg}")
                badn += 1
    print(f"[DONE] 成功 {okn}，失败 {badn}")

if __name__ == "__main__":
    main()
