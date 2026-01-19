#!/usr/bin/env python
"""
Convert Foggy Cityscapes + Cityscapes gtFine polygons to YOLOv5 detection labels (8 classes).

Expected inputs:
- foggy_root: path to leftImg8bit_foggy directory with structure {train,val}/{city}/*.png
- gt_root: path to Cityscapes gtFine directory with structure {train,val}/{city}/*_gtFine_polygons.json

Outputs under out_root (default: datasets/CityScapesFoggy/yolov5_format):
- images/{train,val}/<city>/<image>.png (symlink by default)
- labels/{train,val}/<city>/<image>.txt

Classes (index):
0: bus, 1: bicycle, 2: car, 3: motorcycle, 4: person, 5: rider, 6: train, 7: truck
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

CLASSES = [
    'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck'
]
LABEL_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--foggy-root', type=str, required=True, help='Path to leftImg8bit_foggy root')
    ap.add_argument('--gt-root', type=str, required=True, help='Path to Cityscapes gtFine root')
    ap.add_argument('--out-root', type=str, default='datasets/CityScapesFoggy/yolov5_format', help='Output root for YOLO format')
    ap.add_argument('--splits', type=str, nargs='+', default=['train', 'val'])
    ap.add_argument('--link-images', action='store_true', help='Symlink images instead of copying')
    return ap.parse_args()


def polygon_to_bbox(poly: List[List[int]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x1, y1, x2, y2


def write_yolo_label(label_path: Path, w: int, h: int, objects: List[dict]):
    lines: List[str] = []
    for obj in objects:
        label = obj.get('label')
        if label not in LABEL_TO_ID:
            continue
        poly = obj.get('polygon')
        if not poly or len(poly) < 3:
            continue
        x1, y1, x2, y2 = polygon_to_bbox(poly)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw < 1 or bh < 1:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # normalize
        nx = cx / w
        ny = cy / h
        nw = bw / w
        nh = bh / h
        cls_id = LABEL_TO_ID[label]
        lines.append(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, 'w') as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    foggy_root = Path(args.foggy_root)
    gt_root = Path(args.gt_root)
    out_root = Path(args.out_root)
    img_out_root = out_root / 'images'
    lbl_out_root = out_root / 'labels'

    for split in args.splits:
        foggy_split = foggy_root / split
        gt_split = gt_root / split
        if not foggy_split.exists():
            print(f"[WARN] Missing foggy split: {foggy_split}")
            continue
        for city_dir in sorted([p for p in foggy_split.iterdir() if p.is_dir()]):
            for img_path in city_dir.glob('*.png'):
                # Foggy filename: <city>_<seq>_<frame>_leftImg8bit_foggy_*.png
                stem = img_path.stem
                if '_leftImg8bit_foggy' not in stem:
                    continue
                base = stem.split('_leftImg8bit_foggy')[0]  # <city>_<seq>_<frame>
                gt_json = gt_split / city_dir.name / f"{base}_gtFine_polygons.json"
                if not gt_json.exists():
                    # Some Foggy subsets may include images not in gt; skip
                    continue
                with open(gt_json, 'r') as f:
                    data = json.load(f)
                w, h = data.get('imgWidth'), data.get('imgHeight')
                if not w or not h:
                    # fallback: read image size
                    with Image.open(img_path) as im:
                        w, h = im.size
                # Write label
                rel_dir = Path(split) / city_dir.name
                label_path = lbl_out_root / rel_dir / f"{stem}.txt"
                write_yolo_label(label_path, w, h, data.get('objects', []))
                # Link/copy image
                out_img_path = img_out_root / rel_dir / f"{img_path.name}"
                out_img_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if args.link_images:
                        if out_img_path.exists():
                            out_img_path.unlink()
                        os.symlink(os.path.abspath(img_path), out_img_path)
                    else:
                        if not out_img_path.exists():
                            # hardlink if possible, else copy
                            try:
                                os.link(img_path, out_img_path)
                            except OSError:
                                from shutil import copy2
                                copy2(img_path, out_img_path)
                except Exception as e:
                    print(f"[WARN] Failed to place image {img_path}: {e}")

    print(f"Done. YOLO dataset at: {out_root}")


if __name__ == '__main__':
    main()

