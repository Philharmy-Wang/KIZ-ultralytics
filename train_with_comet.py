from comet_ml import Experiment
from ultralytics import YOLO

# 1) 初始化 Comet，并命名
exp = Experiment(
    api_key="l4yWlRyECB7wmpIGX2t3tcZ8p",
    project_name="project-macaca-detection",
    workspace="philharmy-wang",
    # auto_output_logging="simple",  # 可选：减少噪声
)
# exp.set_name("macaca_yolo11n")          # ← 实验名
# exp.add_tags(["yolo11n", "macaca", "det"])

# 2) 训练
model = YOLO("/mnt/data/wgb/ultralytics/ultralytics/cfg/models/11/yolo11n.yaml")
model.train(
    data="/mnt/data/wgb/ultralytics/datasets/macaca-detection/data.yaml",
    epochs=100, batch=64, imgsz=640, device="0,1",
    project="project/macaca_detection", name="yolo11n"
)
