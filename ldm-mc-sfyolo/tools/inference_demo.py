#!/usr/bin/env python
import argparse
import random
from pathlib import Path
import sys, os
from pathlib import Path as _Path

import torch
import yaml
import numpy as np
from PIL import Image

FILE = _Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.mc_dropout import register_mc_dropout_hooks, aggregate_mc_predictions


@torch.no_grad()
def run(weights: str,
        data_yaml: str,
        imgsz: int,
        device_str: str,
        num_images: int,
        outdir: str,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        mc_T: int = 0,
        mc_p: float = 0.2,
        images: str | None = None):
    device = select_device(device_str)
    model = attempt_load(weights, device=device)
    model.eval()
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    # Build image list: explicit "images" (comma-separated paths) takes precedence
    sel: list[Path]
    if images:
        sel = [Path(p.strip()) for p in images.split(',') if p.strip()]
    else:
        with open(data_yaml, 'r') as f:
            d = yaml.safe_load(f)
        val_dirs = d['val'] if isinstance(d['val'], list) else [d['val']]
        img_dir = Path(d.get('path', '.')) / Path(val_dirs[0])
        all_imgs = list(img_dir.rglob('*.png')) + list(img_dir.rglob('*.jpg'))
        random.seed(0)
        sel = random.sample(all_imgs, min(num_images, len(all_imgs)))

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    for p in sel:
        im = Image.open(p).convert('RGB')
        im_arr = np.array(im)  # RGB
        im_t = torch.from_numpy(im_arr[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0

        if mc_T and mc_T > 0:
            handles = register_mc_dropout_hooks(model, p=mc_p, only_detect=True)
            preds_T = []
            try:
                for _ in range(mc_T):
                    pred, _ = model(im_t)
                    preds_T.append(non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det))
            finally:
                for h in handles:
                    h.remove()
            det_b = aggregate_mc_predictions(preds_T, iou_match=0.6, min_votes=max(2, (mc_T+1)//2), var_thr=1e9)[0]
            if det_b is None or det_b.numel() == 0:
                det = torch.zeros((0, 6), device=device)
            else:
                det = det_b
        else:
            pred, _ = model(im_t)
            det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)[0]
            if det is None:
                det = torch.zeros((0, 6), device=device)

        img = im_arr.copy()
        annotator = Annotator(img, line_width=2)
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                c = int(cls.item())
                label = f"{c}:{conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
        save_path = outdir_p / f"{p.stem}.jpg"
        Image.fromarray(annotator.result()).save(save_path)
    print(f"Saved detections to {outdir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--data', type=str, default='data/foggy_cityscapes.yaml')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--device', type=str, default='')
    ap.add_argument('--num-images', type=int, default=10)
    ap.add_argument('--outdir', type=str, default='runs/vis_detect')
    ap.add_argument('--conf-thres', type=float, default=0.25)
    ap.add_argument('--iou-thres', type=float, default=0.45)
    ap.add_argument('--max-det', type=int, default=50)
    ap.add_argument('--mc-T', type=int, default=0)
    ap.add_argument('--mc-p', type=float, default=0.2)
    ap.add_argument('--images', type=str, default='', help='Comma-separated absolute paths to images')
    args = ap.parse_args()
    run(args.weights, args.data, args.imgsz, args.device, args.num_images, args.outdir,
        args.conf_thres, args.iou_thres, args.max_det, args.mc_T, args.mc_p,
        images=args.images or None)
