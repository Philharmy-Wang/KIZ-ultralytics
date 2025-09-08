#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
目标
----
合并两个 YOLO 检测数据集为统一结构：
- 自动识别两种常见目录结构：
  A) 根目录下存在 train/、valid/（或 val/），每个 split 内有 images/、labels/
  B) 根目录下是 images/、labels/，其内含 train|val(valid)|test 子目录
- 统一输出为：
  OUT_ROOT/
    images/{train,valid,test}
    labels/{train,valid,test}
    data.yaml

关键设计
--------
1) 目录结构自适应：detect_splits(root) 自动产出 split -> (img_dir, lbl_dir)
2) 类别并集：从各自 data.yaml 读取 names（list 或 dict 均可），按“类别名”并集合并 -> merged_names
   - 如两侧 names 完全一致则不重写 id
   - 否则基于名称建立 old_id -> new_id 的映射，并在写出 label 时重写第 1 列 class id
3) 同名冲突：输出位置若已存在同名文件，为该图像与标签文件名添加 TAG 前缀（如 dsA_IMG_001.jpg）
4) 无标签策略：EXCLUDE_UNLABELED=True 时剔除无标签图像（更干净）；False 时保留并写空 txt
5) 放置方式：COPY_MODE = 'symlink' | 'hardlink' | 'copy'，失败自动回退 'copy'
6) 生成新的 data.yaml，直接可用于 Ultralytics 训练

适配你的两套数据：
- SRC1_ROOT=/mnt/data/wgb/ultralytics/datasets/macaca-detection/210_images
  结构：train/images, train/labels, valid/images, valid/labels
- SRC2_ROOT=/mnt/data/wgb/ultralytics/datasets/macaca-detection/yolo_splits/exp1
  结构：images/train, images/val; labels/train, labels/val
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =========================
# 参数区（按需修改）
# =========================

# 源数据集 A（你的 210 张）
SRC1_ROOT = Path("/mnt/data/wgb/ultralytics/datasets/macaca-detection/210_images")
TAG1 = "dsA"  # 用于冲突前缀

# 源数据集 B（另一套按 images/labels 分层的数据集）
SRC2_ROOT = Path("/mnt/data/wgb/ultralytics/datasets/macaca-detection/yolo_splits/exp1")
TAG2 = "dsB"

# 合并输出目录
OUT_ROOT = Path("/mnt/data/wgb/ultralytics/datasets/macaca-detection/merged/210_plus_exp1_v2")

# 验证集输出目录名标准化为 'valid'（输入允许 'val'）
VALID_STD_NAME = "valid"

# 无标签图片是否剔除（推荐 True）
EXCLUDE_UNLABELED = True

# 放置方式：'symlink' | 'hardlink' | 'copy'（失败自动回退 copy）
COPY_MODE = "symlink"

# 类别不一致时是否按“并集”融合（基于类别名）并重写 ID
UNION_NAMES_IF_DIFFERENT = True

# 支持的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================
# 基础工具
# =========================

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_place(src: Path, dst: Path, mode: str):
    """按模式放置文件，失败回退 copy。"""
    safe_mkdir(dst.parent)
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
        shutil.copy2(src, dst)

def list_images(img_dir: Path) -> List[Path]:
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

def next_unique_name(target_dir: Path, filename: str, tag: str) -> str:
    """解决同名冲突：若已存在同名，则加 tag_ 前缀；仍冲突则再加数字递增。"""
    stem = Path(filename).stem
    suf = Path(filename).suffix
    candidate = filename
    if (target_dir / candidate).exists():
        candidate = f"{tag}_{stem}{suf}"
    i = 1
    while (target_dir / candidate).exists():
        candidate = f"{tag}_{stem}_{i}{suf}"
        i += 1
    return candidate


# =========================
# 类别表处理
# =========================

def read_yaml_names(yaml_path: Path) -> Optional[List[str]]:
    """从 data.yaml 读取 names，支持 list 或 {id:name}。读取不到返回 None。"""
    if not yaml_path.exists():
        return None
    import yaml
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not data or "names" not in data:
        return None
    names = data["names"]
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        items = sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])
        return [str(v) for _, v in items]
    return None

def build_name_union(names1: Optional[List[str]], names2: Optional[List[str]]) -> Optional[List[str]]:
    """按顺序合并并去重（基于名称），保持 names1 顺序优先。"""
    if names1 is None and names2 is None:
        return None
    if names1 is None:
        return list(names2)
    if names2 is None:
        return list(names1)
    merged = list(names1)
    for n in names2:
        if n not in merged:
            merged.append(n)
    return merged

def id_mapping_from_names(old_names: Optional[List[str]], new_names: Optional[List[str]]) -> Dict[int, int]:
    """基于名称构造旧 id -> 新 id 映射。若任何一侧为 None，则返回空映射（不改写）。"""
    if old_names is None or new_names is None:
        return {}
    new_index = {n: i for i, n in enumerate(new_names)}
    mapping = {}
    for i, n in enumerate(old_names):
        if n in new_index:
            mapping[i] = new_index[n]
        # 若旧类不在新表，跳过（意味着该类标注会被丢弃）
    return mapping


# =========================
# 目录结构自适应
# =========================

def detect_splits(root: Path) -> Dict[str, Tuple[Path, Path]]:
    """
    返回 split -> (img_dir, lbl_dir) 的映射，split 归一化为：'train'、'valid'、'test'
    兼容两种结构：
      A) root/train/images, root/train/labels; root/valid|val/images, labels; root/test 可选
      B) root/images/train|val|test, root/labels/train|val|test
    """
    m: Dict[str, Tuple[Path, Path]] = {}

    # 结构 A：根下有 train 目录，且存在 train/images, train/labels
    if (root / "train").exists() and (root / "train" / "images").exists() and (root / "train" / "labels").exists():
        m["train"] = (root / "train" / "images", root / "train" / "labels")
        # valid/val
        if (root / "valid").exists() and (root / "valid" / "images").exists() and (root / "valid" / "labels").exists():
            m["valid"] = (root / "valid" / "images", root / "valid" / "labels")
        elif (root / "val").exists() and (root / "val" / "images").exists() and (root / "val" / "labels").exists():
            m["valid"] = (root / "val" / "images", root / "val" / "labels")
        # test
        if (root / "test").exists() and (root / "test" / "images").exists() and (root / "test" / "labels").exists():
            m["test"] = (root / "test" / "images", root / "test" / "labels")
        return m

    # 结构 B：根下有 images 与 labels，且 images/ 内部有 train|val(valid)|test
    if (root / "images").exists() and (root / "labels").exists():
        img_root = root / "images"
        lbl_root = root / "labels"
        # train
        if (img_root / "train").exists() and (lbl_root / "train").exists():
            m["train"] = (img_root / "train", lbl_root / "train")
        # valid|val
        if (img_root / "valid").exists() and (lbl_root / "valid").exists():
            m["valid"] = (img_root / "valid", lbl_root / "valid")
        elif (img_root / "val").exists() and (lbl_root / "val").exists():
            m["valid"] = (img_root / "val", lbl_root / "val")
        # test
        if (img_root / "test").exists() and (lbl_root / "test").exists():
            m["test"] = (img_root / "test", lbl_root / "test")
        if m:
            return m

    raise AssertionError(f"无法识别数据集结构：{root}\n"
                         f"需要以下两种之一：\n"
                         f" A) root/train|valid(/test)/images, labels\n"
                         f" B) root/images/train|val|test, root/labels/train|val|test")

def collect_pairs(img_dir: Path, lbl_dir: Path, exclude_unlabeled: bool) -> List[Tuple[Path, Optional[Path]]]:
    """
    收集 (image, label或None) 列表。
    - 强制以图片为主：按图片 stem 匹配 label
    - exclude_unlabeled=True 时跳过无 label 的图片；否则保留并用空 txt
    """
    imgs = list_images(img_dir)
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for im in imgs:
        lb = lbl_dir / f"{im.stem}.txt"
        if lb.exists():
            pairs.append((im, lb))
        else:
            if exclude_unlabeled:
                continue
            pairs.append((im, None))
    return pairs


# =========================
# 主流程
# =========================

def main():
    # 1) 读取类别表
    names1 = read_yaml_names(SRC1_ROOT / "data.yaml")
    names2 = read_yaml_names(SRC2_ROOT / "data.yaml")

    # 2) 统一类别（完全一致或并集融合）
    if names1 == names2 or (names1 is None and names2 is None):
        merged_names = names1  # 可能为 None
    else:
        if not UNION_NAMES_IF_DIFFERENT:
            raise ValueError(f"两个数据集的类别表不同，请手动统一：\nA: {names1}\nB: {names2}")
        merged_names = build_name_union(names1, names2)

    map1 = id_mapping_from_names(names1, merged_names)
    map2 = id_mapping_from_names(names2, merged_names)

    # 3) 识别两个数据集的 split -> (img_dir, lbl_dir)
    splits1 = detect_splits(SRC1_ROOT)
    splits2 = detect_splits(SRC2_ROOT)

    # 4) 准备输出目录
    out_train_img = OUT_ROOT / "images" / "train"
    out_train_lbl = OUT_ROOT / "labels" / "train"
    out_valid_img = OUT_ROOT / "images" / VALID_STD_NAME
    out_valid_lbl = OUT_ROOT / "labels" / VALID_STD_NAME
    out_test_img  = OUT_ROOT / "images" / "test"
    out_test_lbl  = OUT_ROOT / "labels" / "test"
    for p in [out_train_img, out_train_lbl, out_valid_img, out_valid_lbl]:
        safe_mkdir(p)

    # 5) 合并逻辑（对 train/valid/test 各自处理）
    def merge_split(split_key: str, out_img: Path, out_lbl: Path) -> Tuple[int, int, int]:
        """
        合并指定 split；返回：
        - placed：放置的图像数量
        - skipped_unlabeled：因无标签被跳过的图像数（在 EXCLUDE_UNLABELED=True 时）
        - dropped_anns：被丢弃的标注行数（因类别不在 merged_names）
        """
        placed = 0
        skipped_unlabeled = 0
        dropped_anns = 0

        for (root, tag, s_map) in [
            (SRC1_ROOT, TAG1, splits1),
            (SRC2_ROOT, TAG2, splits2),
        ]:
            if split_key not in s_map:
                continue
            img_dir, lbl_dir = s_map[split_key]
            pairs = collect_pairs(img_dir, lbl_dir, EXCLUDE_UNLABELED)

            # 根据数据集来源选择对应 id 映射
            id_map = map1 if root == SRC1_ROOT else map2
            need_rewrite = not (merged_names is None or (names1 == names2))

            for im, lb in pairs:
                # 解决同名冲突
                new_img_name = next_unique_name(out_img, im.name, tag)
                new_lbl_name = Path(new_img_name).with_suffix(".txt").name

                # 放置图像
                safe_place(im, out_img / new_img_name, COPY_MODE)

                # 放置/改写标签
                dst_lbl = out_lbl / new_lbl_name
                if lb is None:
                    if EXCLUDE_UNLABELED:
                        # 理论上不会到这（collect_pairs 已剔除）；兜底防御
                        skipped_unlabeled += 1
                        try:
                            (out_img / new_img_name).unlink()
                        except Exception:
                            pass
                        continue
                    else:
                        dst_lbl.write_text("", encoding="utf-8")
                        placed += 1
                        continue

                if not need_rewrite or len(id_map) == 0:
                    # 两侧类别表一致，或无可用映射：直接放置
                    safe_place(lb, dst_lbl, COPY_MODE)
                else:
                    # 重写第一列 class id
                    lines = [ln.strip() for ln in lb.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    out_lines = []
                    for ln in lines:
                        parts = ln.split()
                        try:
                            old_id = int(float(parts[0]))
                        except Exception:
                            continue
                        if old_id not in id_map:
                            dropped_anns += 1
                            continue
                        parts[0] = str(id_map[old_id])
                        out_lines.append(" ".join(parts))
                    dst_lbl.write_text("\n".join(out_lines), encoding="utf-8")

                placed += 1

        return placed, skipped_unlabeled, dropped_anns

    total, skipped, dropped = 0, 0, 0

    # train
    a, b, c = merge_split("train", out_train_img, out_train_lbl)
    total += a; skipped += b; dropped += c

    # valid（无论输入是 valid 还是 val，这里统一为 VALID_STD_NAME）
    a, b, c = merge_split("valid", out_valid_img, out_valid_lbl)
    total += a; skipped += b; dropped += c

    # test（若任一数据集存在）
    has_test = ("test" in splits1) or ("test" in splits2)
    if has_test:
        safe_mkdir(out_test_img); safe_mkdir(out_test_lbl)
        a, b, c = merge_split("test", out_test_img, out_test_lbl)
        total += a; skipped += b; dropped += c

    # 6) 写 data.yaml
    def write_data_yaml():
        import yaml
        data = {}
        data["path"]  = str(OUT_ROOT.resolve())
        data["train"] = "images/train"
        data["val"]   = f"images/{VALID_STD_NAME}"
        if has_test:
            data["test"] = "images/test"
        final_names = merged_names or names1 or names2 or ["Macaca"]
        data["names"] = {i: n for i, n in enumerate(final_names)}
        (OUT_ROOT / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return final_names

    final_names = write_data_yaml()

    # 7) 汇总
    def count_files(p: Path) -> int:
        return len([x for x in p.iterdir() if x.is_file()]) if p.exists() else 0

    print("="*72)
    print("合并完成 ✅")
    print(f"输出目录：{OUT_ROOT.resolve()}")
    print(f"train : images={count_files(out_train_img)}  labels={count_files(out_train_lbl)}")
    print(f"valid : images={count_files(out_valid_img)}  labels={count_files(out_valid_lbl)}")
    if has_test:
        print(f"test  : images={count_files(out_test_img)}  labels={count_files(out_test_lbl)}")
    print("-"*72)
    print(f"总放置图像数     : {total}")
    print(f"因无标签被跳过数 : {skipped}  （EXCLUDE_UNLABELED={EXCLUDE_UNLABELED}）")
    print(f"被丢弃标注行数   : {dropped}  （多因类别名不在并集导致）")
    print("-"*72)
    print(f"最终类别表 names : {final_names}")
    print("Ultralytics 训练示例：")
    print(f"yolo detect train data={str((OUT_ROOT/'data.yaml').resolve())} model=yolo11m.pt imgsz=640 epochs=100 batch=16 device=0,1")
    print("="*72)


if __name__ == "__main__":
    main()
