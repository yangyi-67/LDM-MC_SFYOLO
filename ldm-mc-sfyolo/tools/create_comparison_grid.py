#!/usr/bin/env python
import argparse
from pathlib import Path
from PIL import Image, ImageOps

def load_image(path: Path, target_h: int = None):
    img = Image.open(path).convert('RGB')
    if target_h is not None:
        w, h = img.size
        if h != target_h:
            new_w = int(w * (target_h / h))
            img = img.resize((new_w, target_h), Image.BICUBIC)
    return img

def grid3xN(names, col_dirs, out_path: Path, pad: int = 8, bg=(255,255,255)):
    # names: base filenames without suffix (stem), col_dirs: [dir_source, dir_sfyolo, dir_ours]
    cols = len(col_dirs)
    rows = len(names)
    # Load first to determine cell size
    imgs = [[load_image(Path(col_dirs[c]) / f"{names[r]}.jpg") for c in range(cols)] for r in range(rows)]
    # unify heights per row
    max_h = max(im.size[1] for row in imgs for im in row)
    imgs = [[load_image(Path(col_dirs[c]) / f"{names[r]}.jpg", target_h=max_h) for c in range(cols)] for r in range(rows)]
    widths = [max(imgs[r][c].size[0] for r in range(rows)) for c in range(cols)]
    heights = [max(imgs[r][c].size[1] for c in range(cols)) for r in range(rows)]
    total_w = sum(widths) + pad * (cols + 1)
    total_h = sum(heights) + pad * (rows + 1)
    canvas = Image.new('RGB', (total_w, total_h), bg)
    y = pad
    for r in range(rows):
        x = pad
        for c in range(cols):
            im = imgs[r][c]
            # center in cell
            cell_w = widths[c]
            ox = x + (cell_w - im.size[0]) // 2
            canvas.paste(im, (ox, y))
            x += cell_w + pad
        y += heights[r] + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', type=str, required=True, help='Comma-separated base names (without .jpg)')
    ap.add_argument('--col1', type=str, required=True, help='Dir for column 1 (source-only)')
    ap.add_argument('--col2', type=str, required=True, help='Dir for column 2 (sfyolo)')
    ap.add_argument('--col3', type=str, required=True, help='Dir for column 3 (ours)')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    names = [n.strip() for n in args.names.split(',') if n.strip()]
    grid3xN(names, [args.col1, args.col2, args.col3], Path(args.out))

