#!/usr/bin/env bash
set -e

yolo detect train data=fire-smoke.yaml model=yolov5nu.pt  epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolo5n
yolo detect train data=fire-smoke.yaml model=yolov5su.pt  epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolo5s
yolo detect train data=fire-smoke.yaml model=yolo5mu.pt  epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolo5m

yolo detect train data=fire-smoke.yaml model=yolov8n.pt  epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolov8n
yolo detect train data=fire-smoke.yaml model=yolov8s.pt  epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolov8s
yolo detect train data=fire-smoke.yaml model=yolov8m.pt  epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolov8m
yolo detect train data=fire-smoke.yaml model=yolov8l.pt  epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolov8l

yolo detect train data=fire-smoke.yaml model=yolo11s.pt epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolo11s
yolo detect train data=fire-smoke.yaml model=yolo11m.pt epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolo11m
yolo detect train data=fire-smoke.yaml model=yolo11l.pt epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolo11l

yolo detect train data=fire-smoke.yaml model=yolo12n.pt epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolo12n
yolo detect train data=fire-smoke.yaml model=yolo12s.pt epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolo12s
yolo detect train data=fire-smoke.yaml model=yolo12m.pt epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolo12m
yolo detect train data=fire-smoke.yaml model=yolo12l.pt epochs=100  batch=32 imgsz=640 device=0,1 project=project/fire_smoke name=yolo12l
