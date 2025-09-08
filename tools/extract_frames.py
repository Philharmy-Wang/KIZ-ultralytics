import cv2
import math
import argparse
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm   # 新增：进度条库

def time_to_seconds(t):
    """把时间字符串或数字转成秒"""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    parts = [float(p) for p in str(t).split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h*3600 + m*60 + s
    elif len(parts) == 2:
        m, s = parts
        return m*60 + s
    else:
        return float(parts[0])

def main():
    ap = argparse.ArgumentParser(description="Extract frames from a video for detection/tracking annotation.")
    ap.add_argument("-i", "--input", required=True, help="输入视频路径 (.avi/.mp4 等)")
    ap.add_argument("-o", "--output", required=True, help="输出图片目录")
    ap.add_argument("--fps", type=float, default=5.0, help="目标抽帧率（每秒保存多少帧）。<=0 表示保存全部帧")
    ap.add_argument("--every_n", type=int, default=None, help="按帧间隔抽帧：每 N 帧保存 1 帧（优先级高于 --fps）")
    ap.add_argument("--start", type=str, default=None, help="起始时间，比如 00:02:00 或 120（秒）")
    ap.add_argument("--end", type=str, default=None, help="结束时间，比如 00:10:00 或 600（秒）")
    ap.add_argument("--prefix", type=str, default="img", help="输出文件名前缀")
    ap.add_argument("--jpeg", type=int, default=95, help="JPEG质量(1-100)")
    ap.add_argument("--resize_w", type=int, default=None, help="等比例缩放到指定宽度（像素），默认不缩放")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {inp}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_s = time_to_seconds(args.start)
    end_s = time_to_seconds(args.end)

    if start_s is None: start_s = 0.0
    if end_s is None: end_s = total_frames / src_fps
    start_f = max(0, int(math.floor(start_s * src_fps)))
    end_f = min(total_frames, int(math.ceil(end_s * src_fps)))

    # 计算步长
    if args.every_n and args.every_n > 0:
        step = int(args.every_n)
        target_fps = src_fps / step
    else:
        if args.fps is None or args.fps <= 0:
            step = 1
            target_fps = src_fps
        else:
            if args.fps >= src_fps:
                step = 1
                target_fps = src_fps
            else:
                step = max(1, int(round(src_fps / args.fps)))
                target_fps = src_fps / step

    print(f"源FPS={src_fps:.3f}, 目标FPS≈{target_fps:.3f}, 分辨率={width}x{height}, 总帧={total_frames}")
    print(f"抽取范围: {str(timedelta(seconds=start_s))} -> {str(timedelta(seconds=end_s))} | 帧[{start_f}, {end_f}) | 步长={step}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    saved = 0
    frame_id = start_f
    idx = 0

    # 预估需要遍历的总帧数，用于进度条
    total_iter = end_f - start_f

    with tqdm(total=total_iter, desc="抽帧中", unit="帧") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or frame_id >= end_f:
                break

            # 判断是否保存
            save_this = False
            if args.every_n and args.every_n > 0:
                save_this = ((frame_id - start_f) % step == 0)
            else:
                save_this = (idx % step == 0)

            if save_this:
                if args.resize_w and args.resize_w > 0 and args.resize_w != width:
                    scale = args.resize_w / frame.shape[1]
                    new_h = int(round(frame.shape[0] * scale))
                    frame = cv2.resize(frame, (args.resize_w, new_h), interpolation=cv2.INTER_AREA)

                # 用连续编号命名：000001, 000002, ...
                out_path = out_dir / f"{args.prefix}_{saved+1:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg])
                saved += 1

            idx += 1
            frame_id += 1
            pbar.update(1)

    cap.release()
    print(f"完成：输出 {saved} 张到 {out_dir}")

if __name__ == "__main__":
    main()

# python extract_frames.py -i "rec-16328-con-20250410142418-camera-0.avi" -o frames_n30 --every_n 30
