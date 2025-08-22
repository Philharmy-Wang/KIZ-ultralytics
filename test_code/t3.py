from ultralytics import YOLO
from pathlib import Path
import cv2

# 加载模型
model = YOLO("/mnt/data/wgb/ultralytics/runs/pose/train4/weights/best.pt")
results = model("/mnt/data/wgb/ultralytics/demo_img")  # 预测

# 设置保存路径
save_dir = Path("/mnt/data/wgb/ultralytics/outputs/pose_img_nobox")
save_dir.mkdir(parents=True, exist_ok=True)

# 遍历结果
for r in results:
    # 只画关键点与骨架，不画检测框，增大关键点和线条粗细
    im_anno = r.plot(
        boxes=False,          # 不绘制检测框
        kpt_radius=10,         # 关键点半径（增大关键点大小）
        kpt_line=True,        # 绘制骨架
        line_width=3          # 骨架线条粗细
    )
    # 保存结果
    out_path = save_dir / (Path(r.path).stem + "_kpts.png")
    cv2.imwrite(str(out_path), im_anno)