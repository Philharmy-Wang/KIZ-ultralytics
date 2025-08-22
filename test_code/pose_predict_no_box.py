from ultralytics import YOLO
from pathlib import Path
import cv2

model = YOLO("/mnt/data/wgb/ultralytics/runs/pose/train4/weights/best.pt")
results = model("/mnt/data/wgb/ultralytics/demo_img/pic2.png")  # 简写等价于 predict

save_dir = Path("/mnt/data/wgb/ultralytics/outputs/pose_img_nobox")
save_dir.mkdir(parents=True, exist_ok=True)

for r in results:
    # 只画关键点与骨架（不画检测框）
    im_anno = r.plot(boxes=False)      # 返回BGR的numpy图
    out_path = save_dir / (Path(r.path).stem + "_kpts.png")
    cv2.imwrite(str(out_path), im_anno)

    # 如果还要把关键点坐标也存一份：
    # kpts = r.keypoints.xy.cpu().numpy()  # [N, K, 2]
