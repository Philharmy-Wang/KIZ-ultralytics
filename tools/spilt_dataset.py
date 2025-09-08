#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能概述
--------
将 Label Studio 导出的 YOLO 检测数据（images/ 与 labels/）按比例切分为 train/val/(test)，并可生成 Ultralytics 的 data.yaml。
- 严格的图像-标签配对：按文件名 stem 匹配，如 a.jpg ↔ a.txt
- 可配置：输入/输出路径、比例、随机种子、拷贝模式（copy/hardlink/symlink）、是否生成 data.yaml、是否剔除无标签图片
- 健壮性：检查缺失/空标签，输出详细统计报告

编程思路（嵌入式讲解）
--------------------
1) 读取输入根目录下的 images/ 与 labels/，用 Pathlib 收集所有图像文件，允许常见扩展名（.jpg/.png/.jpeg/.bmp/.tif/.tiff）。
2) 通过 stem（不含扩展名的文件基名）构造一一对应的标签路径（stem + '.txt'）。
3) 做一致性校验：
   - 若 exclude_unlabeled=True，则只保留有对应 label 的图像；
   - label 文件空但存在：Ultralytics 也能接受（视为无目标），保留并告警统计。
4) 随机洗牌 + 按比例切分，比例向下取整，剩余向 train 回填，保证总数守恒与复现性（设 SEED）。
5) 输出目录结构：images/{train,val,test} 与 labels/{train,val,test}；根据 copy_mode 执行复制/硬链接/软链接，失败自动回退到 copy。
6) 可选生成 data.yaml（Ultralytics 标准格式），包含 path/train/val/test 与 names。
7) 汇总统计并打印人类可读的报告（总量、缺失、各 split 数量、类别直方图）。

使用方法
--------
python split_yolo_dataset.py
或作为模块导入后调用 main()。
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Dict

# =========================
# 可配置参数（按需改这里）
# =========================

# 输入：Label Studio 导出的 YOLO 目录（内含 images/ 与 labels/）
INPUT_ROOT = Path("/mnt/data/wgb/ultralytics/datasets/macaca-detection/label_studio/1st_label/origin")

# 输出：划分后的 YOLO 目录（会自动创建子目录）
OUTPUT_ROOT = Path("/mnt/data/wgb/ultralytics/datasets/macaca-detection/yolo_splits/exp1")

# 划分比例：三者和应 <= 1.0；若 test=0 则不生成 test 子集
SPLIT_RATIOS = {
    "train": 0.8,
    "val":   0.2,
    "test":  0.0,
}

# 随机种子：确保每次划分一致
SEED = 2025

# 是否剔除没有 label 的图片（推荐 True）
EXCLUDE_UNLABELED = True

# 拷贝模式：'copy' | 'hardlink' | 'symlink'
COPY_MODE = "symlink"

# 是否生成 Ultralytics data.yaml
WRITE_YAML = True

# YAML 中的类别名称（按你的数据集实际类别顺序填写）
NAMES = ["Macaca"]

# 允许的图片扩展名（全部小写）
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================
# 实用函数
# =========================

def ensure_dir(p: Path):
    """若目录不存在则创建。"""
    p.mkdir(parents=True, exist_ok=True)

def list_images(img_dir: Path) -> List[Path]:
    """列出 img_dir 下所有允许扩展名的图片文件（不递归）。"""
    files = []
    for x in img_dir.iterdir():
        if x.is_file() and x.suffix.lower() in IMG_EXTS:
            files.append(x)
    return sorted(files)

def pair_label(img_path: Path, labels_dir: Path) -> Path:
    """根据图片 stem 构造对应 label 路径（.txt）。"""
    return labels_dir / f"{img_path.stem}.txt"

def read_label_stats(label_path: Path) -> Tuple[int, Dict[int, int]]:
    """
    读取 YOLO 标签文件，返回：
    - 标注行数（目标数）
    - 类别直方图 {class_id: count}
    若文件不存在或为空，返回(0, {})
    """
    if not label_path.exists():
        return 0, {}
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0, {}

    cls_hist = {}
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        parts = ln.split()
        try:
            cls_id = int(float(parts[0]))  # 有些导出可能是 "0" 或 "0.0"
            cls_hist[cls_id] = cls_hist.get(cls_id, 0) + 1
        except Exception:
            # 容错：若某行格式异常，跳过
            continue
    return len(lines), cls_hist

def safe_link_or_copy(src: Path, dst: Path, mode: str):
    """
    将 src 安置到 dst（文件层面），支持：
    - symlink：软链接（优先节省磁盘）
    - hardlink：硬链接（同一文件系统内节省磁盘）
    - copy：复制（最稳妥）
    失败自动回退到 copy。
    """
    if dst.exists():
        return
    try:
        if mode == "symlink":
            os.symlink(src, dst)
        elif mode == "hardlink":
            os.link(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown copy mode: {mode}")
    except Exception:
        # 回退到 copy
        shutil.copy2(src, dst)

def split_indices(n: int, ratios: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    """
    将 [0..n-1] 随机划分为 train/val/test 索引集合。
    - 向下取整分配，余数全部补到 train，保证总数守恒
    """
    rnd = random.Random(seed)
    idxs = list(range(n))
    rnd.shuffle(idxs)

    n_train = int(n * ratios.get("train", 0.0))
    n_val   = int(n * ratios.get("val", 0.0))
    n_test  = int(n * ratios.get("test", 0.0))

    # 余数补到 train
    used = n_train + n_val + n_test
    if used < n:
        n_train += (n - used)

    splits = {
        "train": idxs[:n_train],
        "val":   idxs[n_train:n_train+n_val],
        "test":  idxs[n_train+n_val:n_train+n_val+n_test] if n_test > 0 else [],
    }
    return splits

def write_ultra_yaml(out_root: Path, names: List[str], has_test: bool):
    """
    生成 Ultralytics 的 data.yaml。路径采用相对 out_root 的子路径，便于移植。
    """
    yaml_path = out_root / "data.yaml"
    lines = []
    lines.append(f"path: {out_root.resolve()}")
    lines.append(f"train: images/train")
    lines.append(f"val: images/val")
    if has_test:
        lines.append(f"test: images/test")
    lines.append("names:")
    for i, nm in enumerate(names):
        lines.append(f"  {i}: {nm}")
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path

# =========================
# 主流程
# =========================

def main():
    images_dir = INPUT_ROOT / "images"
    labels_dir = INPUT_ROOT / "labels"

    assert images_dir.exists(), f"未找到 images 目录：{images_dir}"
    assert labels_dir.exists(), f"未找到 labels 目录：{labels_dir}"

    # 1) 收集图片并配对标签
    imgs = list_images(images_dir)

    pairs = []             # (img_path, lbl_path)
    missing_labels = []    # 无对应 label 的图片
    empty_labels = 0       # 空标签文件的数量（存在但无任何行）
    cls_hist_global = {}   # 全局类别直方图（按目标计数）

    for im in imgs:
        lb = pair_label(im, labels_dir)
        if not lb.exists():
            if EXCLUDE_UNLABELED:
                missing_labels.append(im)
                continue
            else:
                # 允许无标签图片：Ultralytics 可训练但会当作背景
                pairs.append((im, lb))
                continue

        # 统计标签
        n_objs, cls_hist = read_label_stats(lb)
        if n_objs == 0:
            empty_labels += 1
        for k, v in cls_hist.items():
            cls_hist_global[k] = cls_hist_global.get(k, 0) + v

        pairs.append((im, lb))

    total_imgs = len(imgs)
    total_pairs = len(pairs)

    # 2) 划分索引
    splits = split_indices(total_pairs, SPLIT_RATIOS, SEED)

    # 3) 准备输出目录
    for sub in ["images/train", "labels/train", "images/val", "labels/val"]:
        ensure_dir(OUTPUT_ROOT / sub)
    has_test = len(splits["test"]) > 0
    if has_test:
        ensure_dir(OUTPUT_ROOT / "images/test")
        ensure_dir(OUTPUT_ROOT / "labels/test")

    # 4) 执行分发
    def place(pair_list: List[Tuple[Path, Path]], idxs: List[int], split_name: str):
        for i in idxs:
            im, lb = pair_list[i]
            # 目标路径
            dst_im = OUTPUT_ROOT / f"images/{split_name}" / im.name
            dst_lb = OUTPUT_ROOT / f"labels/{split_name}" / (im.stem + ".txt")
            # 放置图片
            safe_link_or_copy(im, dst_im, COPY_MODE)
            # 放置标签：若不存在（保留无标签模式下），写空文件
            if lb.exists():
                safe_link_or_copy(lb, dst_lb, COPY_MODE)
            else:
                # 明确写空 txt，保持文件结构完整
                dst_lb.write_text("", encoding="utf-8")

    place(pairs, splits["train"], "train")
    place(pairs, splits["val"],   "val")
    if has_test:
        place(pairs, splits["test"],  "test")

    # 5) 写 data.yaml（可选）
    yaml_path = None
    if WRITE_YAML:
        yaml_path = write_ultra_yaml(OUTPUT_ROOT, NAMES, has_test)

    # 6) 打印报告
    print("="*60)
    print("YOLO 数据集划分完成")
    print(f"输入 images 总数         : {total_imgs}")
    print(f"有效成对（用于划分）数量 : {total_pairs}")
    if EXCLUDE_UNLABELED:
        print(f"因缺少 label 被剔除的图片: {len(missing_labels)}")
    else:
        print(f"允许无标签图片，缺少 label 的图片: {len([im for im in imgs if not (labels_dir / (im.stem + '.txt')).exists()])}")
    print(f"空标签文件数量（存在但0行）: {empty_labels}")
    print("-"*60)
    print("各 Split 数量：")
    print(f"  train: {len(splits['train'])}")
    print(f"  val  : {len(splits['val'])}")
    if has_test:
        print(f"  test : {len(splits['test'])}")
    print("-"*60)
    print("类别直方图（按目标计数）:")
    if cls_hist_global:
        for k in sorted(cls_hist_global.keys()):
            print(f"  class {k}: {cls_hist_global[k]}")
    else:
        print("  未统计到目标（可能大部分为空标签）。")
    print("-"*60)
    print(f"输出目录：{OUTPUT_ROOT.resolve()}")
    if yaml_path:
        print(f"data.yaml：{yaml_path.resolve()}")
        print("\nUltralytics 训练示例：")
        print(f"yolo detect train data={yaml_path.resolve()} model=yolo11m.pt imgsz=640 epochs=100 batch=16 device=0,1")
    print("="*60)

if __name__ == "__main__":
    main()
