from ultralytics import YOLO
from pathlib import Path
import numpy as np
import csv
import cv2

# =========================
# 路径配置
# =========================
model_path = "/mnt/data/wgb/ultralytics/runs/pose/train4/weights/last.pt"  # 你的狗姿态模型
img_path   = "/mnt/data/wgb/ultralytics/demo_img/pic3.png"
save_root  = Path("/mnt/data/wgb/ultralytics/outputs/pose_tuned2")
save_root.mkdir(parents=True, exist_ok=True)

# =========================
# 推理参数网格（由“更宽松”到“略收紧”）
# =========================
param_grid = [
    {"conf": 0.05, "iou": 0.70, "imgsz": 960,  "max_det": 50},
    {"conf": 0.10, "iou": 0.70, "imgsz": 1280, "max_det": 100},
    {"conf": 0.15, "iou": 0.75, "imgsz": 1280, "max_det": 100},
]
KPT_CONF_THR = 0.20  # 二次阈值：关键点层面的可视化置信度阈值

# =========================
# 骨架连线模板
# =========================
COCO17_EDGES = [
    (5,7),(7,9), (6,8),(8,10),            # 双臂
    (11,13),(13,15), (12,14),(14,16),     # 双腿
    (5,6), (5,11),(6,12),                 # 躯干
    (0,1),(0,2),(1,3),(2,4)               # 头部（鼻-眼-耳）
]

# 猜测版 Dog 24 点骨架；运行后看 *_kpts_idx.png 的编号再微调：
DOG24_EDGES_GUESS = [
    (0,1),(0,2),(1,3),(2,4),      # 鼻-眼-耳
    (0,5),(5,6),(6,7),(7,8),      # 鼻-颈-肩胛-背中-腰/骶
    (8,9),(9,10),                 # 尾巴（若你的标注只有尾基/尾尖，保留一段即可）
    (6,11),(11,12),(12,13),       # 左前肢
    (6,14),(14,15),(15,16),       # 右前肢
    (8,17),(17,18),(18,19),       # 左后肢
    (8,20),(20,21),(21,22),       # 右后肢
    (11,14),(17,20)               # 横向稳定
]

def draw_kpts(im_bgr, kpts_xy, kpts_conf, edges, kpt_thr=0.2, draw_index=False, draw_edges=True):
    """
    只画关键点与骨架（不画框）。
    - draw_index=True 时，给每个关键点标注索引编号
    - draw_edges=False 时，仅画点/编号，不画连线
    """
    im = im_bgr.copy()

    # 画点 + 编号
    for j, (x, y) in enumerate(kpts_xy):
        if np.isnan(x) or np.isnan(y):
            continue
        c = float(kpts_conf[j]) if kpts_conf is not None else 1.0
        if c < kpt_thr:
            continue
        cv2.circle(im, (int(x), int(y)), 3, (255, 255, 255), -1)
        cv2.circle(im, (int(x), int(y)), 2, (0, 0, 0), -1)
        if draw_index:
            # 小白底 + 黑字，便于看清编号
            cv2.rectangle(im, (int(x)+5, int(y)-12), (int(x)+5+12, int(y)+2), (255,255,255), -1)
            cv2.putText(im, str(j), (int(x)+6, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

    # 画骨架
    if draw_edges and edges:
        for a, b in edges:
            if a < len(kpts_xy) and b < len(kpts_xy):
                x1, y1 = kpts_xy[a]
                x2, y2 = kpts_xy[b]
                if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                    continue
                c1 = float(kpts_conf[a]) if kpts_conf is not None else 1.0
                c2 = float(kpts_conf[b]) if kpts_conf is not None else 1.0
                if c1 >= kpt_thr and c2 >= kpt_thr:
                    cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                    cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1)
    return im

def save_kpts_csv(csv_path, all_instances_xy, all_instances_conf):
    """保存 CSV：instance_id, kp_id, x, y, conf"""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "keypoint_id", "x", "y", "confidence"])
        for i, (xy, cf) in enumerate(zip(all_instances_xy, all_instances_conf)):
            for j, (x, y) in enumerate(xy):
                c = float(cf[j]) if cf is not None else 1.0
                w.writerow([i, j, float(x), float(y), c])

# =========================
# 加载模型并推理
# =========================
model = YOLO(model_path)

for gi, pars in enumerate(param_grid):
    results = model.predict(
        source=img_path,
        conf=pars["conf"],
        iou=pars["iou"],
        imgsz=pars["imgsz"],
        max_det=pars["max_det"],
        verbose=False,
    )

    for r in results:
        im0 = cv2.imread(r.path)
        stem = Path(r.path).stem

        # 自动识别关键点数量K，用以选择骨架
        K = int(r.keypoints.xy.shape[2]) if r.keypoints.xy.ndim == 3 else r.keypoints.xy.shape[1]
        if K == 17:
            edges = COCO17_EDGES
        elif K == 24:
            edges = DOG24_EDGES_GUESS
        else:
            edges = []  # 未知K：先不连线，等你确认编号后再自定义

        all_xy, all_cf = [], []
        for inst_id in range(len(r.keypoints)):
            kxy  = r.keypoints.xy[inst_id].cpu().numpy()     # [K,2]
            if getattr(r.keypoints, "conf", None) is not None:
                kcnf = r.keypoints.conf[inst_id].cpu().numpy()  # [K]
            else:
                kcnf = np.ones((kxy.shape[0],), dtype=np.float32)

            all_xy.append(kxy)
            all_cf.append(kcnf)

        out_dir = save_root / f"g{gi}_conf{pars['conf']}_iou{pars['iou']}_sz{pars['imgsz']}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) 默认可视化：画点+连线（不画框）
        im_kpts = im0.copy()
        for kxy, kcnf in zip(all_xy, all_cf):
            im_kpts = draw_kpts(im_kpts, kxy, kcnf, edges, kpt_thr=KPT_CONF_THR,
                                draw_index=False, draw_edges=True)
        cv2.imwrite(str(out_dir / f"{stem}_kpts.png"), im_kpts)

        # 2) 编号可视化：画点+编号+连线（方便你核对索引与骨架是否匹配）
        im_idx_edges = im0.copy()
        for kxy, kcnf in zip(all_xy, all_cf):
            im_idx_edges = draw_kpts(im_idx_edges, kxy, kcnf, edges, kpt_thr=0.0,
                                     draw_index=True, draw_edges=True)
        cv2.imwrite(str(out_dir / f"{stem}_kpts_idx_edges.png"), im_idx_edges)

        # 3) 导出CSV（坐标+置信度）
        save_kpts_csv(out_dir / f"{stem}_kpts.csv", all_xy, all_cf)

        print(f"[Saved] {out_dir}/{stem}_kpts.png, {stem}_kpts_idx_edges.png, {stem}_kpts.csv")

print(f"全部输出在：{save_root}")
