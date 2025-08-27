# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

# ========== 1) 路径 ==========
model = YOLO("/mnt/data/wgb/ultralytics/runs/pose/train4/weights/last.pt")
source = "/mnt/data/wgb/ultralytics/demo_img/pic3.png"
save_dir = Path("/mnt/data/wgb/ultralytics/outputs/pose_img_nobox")
save_dir.mkdir(parents=True, exist_ok=True)

# ========== 2) 定义骨架（Dog 24点；如与你的点位顺序不同，按编号图微调）==========
EDGES_DOG24 = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # 鼻-眼-耳
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # 鼻-颈-肩胛-背中-腰/骶
    (8, 9),
    (9, 10),  # 尾巴（若只有尾基/尾尖就保留一段）
    (6, 11),
    (11, 12),
    (12, 13),  # 左前肢
    (6, 14),
    (14, 15),
    (15, 16),  # 右前肢
    (8, 17),
    (17, 18),
    (18, 19),  # 左后肢
    (8, 20),
    (20, 21),
    (21, 22),  # 右后肢
    (11, 14),
    (17, 20),  # 横向稳定
]

# （可选）COCO 17 点
EDGES_COCO17 = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6),
    (5, 11),
    (6, 12),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

KPT_CONF_THR = 0.20  # 只画高于该置信度的点与边


def draw_points_and_skeleton(img_bgr, kpts_xy, kpts_conf=None, edges=None, thr=0.2, draw_index=False):
    """在 img_bgr 上绘制关键点与骨架连线（不画检测框）."""
    im = img_bgr.copy()

    # 画点和编号
    for j, (x, y) in enumerate(kpts_xy):
        if np.isnan(x) or np.isnan(y):
            continue
        c = float(kpts_conf[j]) if kpts_conf is not None else 1.0
        if c < thr:
            continue
        cv2.circle(im, (int(x), int(y)), 3, (255, 255, 255), -1)
        cv2.circle(im, (int(x), int(y)), 2, (0, 0, 0), -1)
        if draw_index:
            cv2.rectangle(im, (int(x) + 5, int(y) - 12), (int(x) + 5 + 12, int(y) + 2), (255, 255, 255), -1)
            cv2.putText(im, str(j), (int(x) + 6, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # 画骨架
    if edges:
        for a, b in edges:
            if a < len(kpts_xy) and b < len(kpts_xy):
                x1, y1 = kpts_xy[a]
                x2, y2 = kpts_xy[b]
                if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                    continue
                c1 = float(kpts_conf[a]) if kpts_conf is not None else 1.0
                c2 = float(kpts_conf[b]) if kpts_conf is not None else 1.0
                if c1 >= thr and c2 >= thr:
                    cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                    cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1)
    return im


# ========== 3) 推理 ==========
results = model(source)

for r in results:
    im0 = cv2.imread(r.path)
    stem = Path(r.path).stem

    # ✅ 正确获取 K（关键点数量）：[N, K, 2] → 取 shape[1]
    K = r.keypoints.xy.shape[1]

    # 选择骨架
    if K == 24:
        edges = EDGES_DOG24
    elif K == 17:
        edges = EDGES_COCO17
    else:
        edges = []  # 未知K：先只画点；确认编号顺序后再自定义 edges

    # 画到同一张图；同时输出“编号校对版”
    im_kpts = im0.copy()
    im_kpts_idx = im0.copy()

    for i in range(len(r.keypoints)):
        kxy = r.keypoints.xy[i].cpu().numpy()  # [K,2]
        kcnf = (
            r.keypoints.conf[i].cpu().numpy()
            if getattr(r.keypoints, "conf", None) is not None
            else np.ones((K,), dtype=np.float32)
        )

        im_kpts = draw_points_and_skeleton(im_kpts, kxy, kcnf, edges, thr=KPT_CONF_THR, draw_index=False)
        im_kpts_idx = draw_points_and_skeleton(im_kpts_idx, kxy, kcnf, edges, thr=0.0, draw_index=True)

    # 保存
    out_path_clean = save_dir / f"{stem}_kpts.png"
    out_path_idx = save_dir / f"{stem}_kpts_idx_edges.png"
    cv2.imwrite(str(out_path_clean), im_kpts)
    cv2.imwrite(str(out_path_idx), im_kpts_idx)

    print(f"[Saved] {out_path_clean}")
    print(f"[Saved] {out_path_idx}")
