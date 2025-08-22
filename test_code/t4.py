from ultralytics import YOLO
from pathlib import Path
import cv2
import time
import numpy as np

# 加载模型
model = YOLO("/mnt/data/wgb/ultralytics/runs/pose/train4/weights/best.pt")
results = model("/mnt/data/wgb/ultralytics/demo_img/")  # 预测

# 设置保存路径
save_dir = Path("/mnt/data/wgb/ultralytics/outputs/pose_img_nobox")
save_dir.mkdir(parents=True, exist_ok=True)

# 获取当前时间戳用于命名
timestamp = time.strftime("%Y%m%d_%H%M%S")

# COCO 数据集的骨架拓扑（关键点之间的连接，基于 17 个关键点）
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # 头部
    (5, 7), (7, 9), (6, 8), (8, 10),                # 手臂
    (5, 11), (6, 12),                               # 躯干到髋部
    (11, 13), (13, 15), (12, 14), (14, 16)          # 腿部
]

# 遍历结果
for i, r in enumerate(results):
    # 获取图像和关键点数据
    img = r.orig_img.copy()  # BGR 格式的原始图像
    keypoints = r.keypoints.data.cpu().numpy()  # 关键点数据 [num_instances, num_keypoints, 3]

    # 手动绘制关键点和骨架
    for instance in keypoints:
        # 绘制关键点
        for kpt in instance:
            x, y, conf = kpt
            if conf > 0.5:  # 仅绘制置信度大于 0.5 的关键点
                cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)

        # 绘制骨架
        for start, end in COCO_SKELETON:
            if instance[start][2] > 0.5 and instance[end][2] > 0.5:  # 确保两个关键点都有效
                start_x, start_y = int(instance[start][0]), int(instance[start][1])
                end_x, end_y = int(instance[end][0]), int(instance[end][1])
                cv2.line(img, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=3)

    # 保存结果
    out_path = save_dir / f"{Path(r.path).stem}_kpts_skeleton_{timestamp}_{i+1}.png"
    cv2.imwrite(str(out_path), img)
    print(f"Saved image with keypoints and skeleton to {out_path}")

    # 调试信息：打印关键点数量
    print(f"Number of detected keypoints in instance {i+1}: {len(keypoints)}")