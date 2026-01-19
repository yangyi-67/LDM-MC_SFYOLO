#!/usr/bin/env python
import argparse
import random
from pathlib import Path

import torch
import yaml
from PIL import Image
import numpy as np
import sys, os
from pathlib import Path as _Path
FILE = _Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = _Path(os.path.relpath(ROOT, _Path.cwd()))

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, xyxy2xywhn
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from TargetAugment.enhance_style import get_style_images
from TargetAugment.enhance_vgg16 import enhance_vgg16
from TargetAugment.enhance_ldm import get_ldm_images


@torch.no_grad()
def run(weights: str,
        data_yaml: str,
        imgsz: int = 960,
        device_str: str = '',
        num_images: int = 10,
        outdir: str = 'runs/vis_compare',
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 50,
        tam_decoder: str = '',
        tam_encoder: str = '',
        tam_fc1: str = '',
        tam_fc2: str = '',
        style_path: str = '',
        style_alpha: float = 1.0):
    device = select_device(device_str)
    model = attempt_load(weights, device=device)
    model.eval()
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    with open(data_yaml, 'r') as f:
        d = yaml.safe_load(f)
    # pick val dir
    val_dirs = d['val'] if isinstance(d['val'], list) else [d['val']]
    root_path = d.get('path', '.')
    img_dir = Path(root_path) / Path(val_dirs[0])
    all_imgs = list(img_dir.rglob('*.png')) + list(img_dir.rglob('*.jpg'))
    random.seed(0)
    sel = random.sample(all_imgs, min(num_images, len(all_imgs)))

    use_tam = all([tam_decoder, tam_encoder, tam_fc1, tam_fc2])
    adain = None
    if use_tam:
        try:
            adain = enhance_vgg16(argparse.Namespace(
                imgsz=imgsz,
                cuda=True if device.type != 'cpu' else False,
                random_style=(style_path == ''),
                style_add_alpha=style_alpha,
                save_style_samples=False,
                style_path=style_path,
                imgs_paths=[],
                decoder_path=tam_decoder,
                encoder_path=tam_encoder,
                fc1=tam_fc1,
                fc2=tam_fc2,
                log_dir='runs/vis_compare'
            ))
        except Exception as e:
            print(f"[WARN] Failed to initialize TAM ({e}). Continuing with LDM only.")
            use_tam = False

    outdir = Path(outdir)
    if use_tam:
        (outdir / 'tam').mkdir(parents=True, exist_ok=True)
    (outdir / 'ldm').mkdir(parents=True, exist_ok=True)

    for p in sel:
        im = Image.open(p).convert('RGB')
        im_arr = np.array(im)[:, :, ::-1].copy()  # to BGR with positive stride
        im_t = torch.from_numpy(im_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        im_255 = im_t.to(torch.float32)  # TAM expects float (0..255)

        # TAM
        if use_tam:
            tam = get_style_images(im_255.clone(), argparse.Namespace(
                imgsz=imgsz,
                cuda=True if device.type != 'cpu' else False,
                random_style=(style_path == ''),
                style_add_alpha=style_alpha,
                save_style_samples=False,
                style_path=style_path,
                imgs_paths=[str(p)]
            ), adain=adain)
        # LDM
        ldm = get_ldm_images(im_t.clone(), argparse.Namespace(
            ldm_model='stabilityai/stable-diffusion-2-1',
            ldm_prompt='foggy city street, dense fog, haze, realistic',
            ldm_negative_prompt='clear sky, sunny, no fog, haze-free',
            ldm_strength=0.35,
            ldm_guidance_scale=5.0,
            ldm_steps=10,
            ldm_seed=-1,
            ldm_prob=1.0,
            ldm_cache_dir=''  # disable cache for quick vis
        ))

        pairs = [('ldm', ldm)]
        if use_tam:
            pairs = [('tam', tam)] + pairs
        for tag, x in pairs:
            x = x.to(device).float() / 255
            pred, _ = model(x)
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)[0]
            img_bgr = (x[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_rgb = img_bgr[:, :, ::-1].copy()
            annotator = Annotator(img_rgb, line_width=2)
            if pred is not None and len(pred):
                for *xyxy, conf, cls in pred:
                    c = int(cls.item())
                    label = f"{c}:{conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
            save_path = outdir / tag / f"{p.stem}.jpg"
            Image.fromarray(annotator.result()).save(save_path)
    print(f"Saved TAM and LDM visualizations to {outdir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--data', type=str, default='data/foggy_cityscapes.yaml')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--device', type=str, default='')
    ap.add_argument('--num-images', type=int, default=10)
    ap.add_argument('--outdir', type=str, default='runs/vis_compare')
    ap.add_argument('--conf-thres', type=float, default=0.25)
    ap.add_argument('--iou-thres', type=float, default=0.45)
    ap.add_argument('--max-det', type=int, default=50)
    ap.add_argument('--tam-decoder', type=str, default='')
    ap.add_argument('--tam-encoder', type=str, default='')
    ap.add_argument('--tam-fc1', type=str, default='')
    ap.add_argument('--tam-fc2', type=str, default='')
    ap.add_argument('--style-path', type=str, default='')
    ap.add_argument('--style-alpha', type=float, default=1.0)
    args = ap.parse_args()
    run(weights=args.weights,
        data_yaml=args.data,
        imgsz=args.imgsz,
        device_str=args.device,
        num_images=args.num_images,
        outdir=args.outdir,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        tam_decoder=args.tam_decoder,
        tam_encoder=args.tam_encoder,
        tam_fc1=args.tam_fc1,
        tam_fc2=args.tam_fc2,
        style_path=args.style_path,
        style_alpha=args.style_alpha)
