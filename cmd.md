# CLI
**Detection Demo**
yolo detect train data=coco.yaml model=yolo11n.pt epochs=100 batch=256 imgsz=640 device=0,1 projetc=test_train name=coco_train

**Detection fire smoke**
yolo detect train data=fire-smoke.yaml model=yolo5n.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo5n

yolo detect train data=fire-smoke.yaml model=yolo5s.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo5s

yolo detect train data=fire-smoke.yaml model=yolo5m.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo5m

yolo detect train data=fire-smoke.yaml model=yolo8n.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo8n

yolo detect train data=fire-smoke.yaml model=yolo8s.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo8s

yolo detect train data=fire-smoke.yaml model=yolo8m.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo8m

yolo detect train data=fire-smoke.yaml model=yolo11n.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo11n

yolo detect train data=fire-smoke.yaml model=yolo11s.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo11s

yolo detect train data=fire-smoke.yaml model=yolo11m.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo11m

yolo detect train data=fire-smoke.yaml model=yolo11l.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo11l

yolo detect train data=fire-smoke.yaml model=yolo12n.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo12n

yolo detect train data=fire-smoke.yaml model=yolo12s.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo12s

yolo detect train data=fire-smoke.yaml model=yolo12m.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo12m

yolo detect train data=fire-smoke.yaml model=yolo12l.pt epochs=100 batch=256 imgsz=640 device=0,1 project=project/fire_smoke name=yolo12l

**Detection macaca**
export COMET_EXPERIMENT_NAME="macaca_yolo11n"
yolo detect train data=/mnt/data/wgb/ultralytics/datasets/macaca-detection/data.yaml model=/mnt/data/wgb/ultralytics/ultralytics/cfg/models/11/yolo11n.yaml epochs=100 batch=64 imgsz=640 device=0,1 project=project/macaca_detection name=yolo11n 

export COMET_EXPERIMENT_NAME="macaca_yolo11s"
yolo detect train data=/mnt/data/wgb/ultralytics/datasets/macaca-detection/data.yaml model=/mnt/data/wgb/ultralytics/ultralytics/cfg/models/11/yolo11s.yaml epochs=100 batch=64 imgsz=640 device=0,1 project=project/macaca_detection name=yolo11s

export COMET_EXPERIMENT_NAME="macaca_yolo11m"
yolo detect train data=/mnt/data/wgb/ultralytics/datasets/macaca-detection/data.yaml model=/mnt/data/wgb/ultralytics/ultralytics/cfg/models/11/yolo11m.yaml epochs=100 batch=32 imgsz=640 device=0,1 project=project/macaca_detection name=yolo11m

export COMET_EXPERIMENT_NAME="macaca_yolo11l"
yolo detect train data=/mnt/data/wgb/ultralytics/datasets/macaca-detection/data.yaml model=/mnt/data/wgb/ultralytics/ultralytics/cfg/models/11/yolo11l.yaml epochs=100 batch=32 imgsz=640 device=0,1 project=project/macaca_detection name=yolo11l

export COMET_EXPERIMENT_NAME="rt_detr-l"
yolo detect train data=/mnt/data/wgb/ultralytics/datasets/macaca-detection/data.yaml model=/mnt/data/wgb/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml epochs=100 batch=32 imgsz=640 device=0,1 project=project/macaca_detection name=rt_detr-l

export COMET_EXPERIMENT_NAME="rtdetr-resnet50"
yolo detect train data=/mnt/data/wgb/ultralytics/datasets/macaca-detection/data.yaml model=/mnt/data/wgb/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml epochs=100 batch=32 imgsz=640 device=0,1 project=project/macaca_detection name=rtdetr-resnet50

export COMET_EXPERIMENT_NAME="rtdetr-resnet101"
yolo detect train data=/mnt/data/wgb/ultralytics/datasets/macaca-detection/data.yaml model=/mnt/data/wgb/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-resnet101.yaml epochs=100 batch=32 imgsz=640 device=0,1 project=project/macaca_detection name=rtdetr-resnet101


**Pose**
yolo pose train data=coco-pose.yaml model=yolo11n-pose.pt epochs=100 batch=256 imgsz=640 device=0,1 project=test_train name=coco-pose_train

yolo pose train data=tiger-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640 device=0,1 project=test_train name=tiger-pose_train

yolo pose train data=dog-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640 device=0,1 project=test_train name=dog-pose_train

预测
yolo pose predict model=yolo11l-pose.pt source='demo_img/pic1.png'
yolo pose predict model=/mnt/data/wgb/ultralytics/runs/pose/train4/weights/best.pt source='/mnt/data/wgb/ultralytics/demo_img/' show=True save=True

yolo pose predict model=/mnt/data/wgb/ultralytics/runs/pose/train4/weights/best.pt source='/mnt/data/wgb/ultralytics/demo_img/dog/' show=True save=True

yolo pose predict model=yolo11l-pose.pt source='/mnt/data/wgb/ultralytics/demo_img/dog/' show=True save=True