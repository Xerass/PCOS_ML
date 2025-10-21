from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import shutil

SRC = Path("data")                 # data/<class>/<image>
DST = Path("data_sanitized")       # mirror will be written here
DST.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 384                     # common working size (we’ll still resize to 224 at train time)

def trim_black_border(img, threshold=8, margin=4, min_keep=64):
    g = np.array(img.convert("L"))
    mask = g > threshold
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    y1 = min(g.shape[0]-1, y1 + margin); x1 = min(g.shape[1]-1, x1 + margin)
    if (y1 - y0 + 1) < min_keep or (x1 - x0 + 1) < min_keep:
        return img
    return img.crop((x0, y0, x1+1, y1+1))

def center_crop_percent(img, pct=0.85):
    # crop a fixed fraction of the shorter side (kills borders/text at edges)
    w, h = img.size
    side = int(min(w, h) * pct)
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - side // 2); y0 = max(0, cy - side // 2)
    x1 = x0 + side; y1 = y0 + side
    return img.crop((x0, y0, x1, y1))

def to_gray3(img):
    g = img.convert("L")
    return Image.merge("RGB", (g, g, g))

def sanitize_one(src_path: Path, dst_path: Path, inner_crop_pct=0.85):
    img = Image.open(src_path).convert("RGB")
    img = trim_black_border(img, threshold=8, margin=4)
    img = center_crop_percent(img, pct=inner_crop_pct)
    img = ImageOps.equalize(img)                     # deterministic equalization
    img = to_gray3(img)
    img = ImageOps.contain(img, (IMG_SIZE, IMG_SIZE))   # keep aspect while fitting box
    # pad to square IMG_SIZE x IMG_SIZE so geometry is identical
    bg = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))
    bg.paste(img, ((IMG_SIZE - img.size[0]) // 2, (IMG_SIZE - img.size[1]) // 2))
    bg.save(dst_path)

# Mirror folder
for cls_dir in sorted([p for p in SRC.iterdir() if p.is_dir()]):
    out_dir = DST / cls_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in cls_dir.iterdir():
        if p.is_file():
            try:
                sanitize_one(p, out_dir / p.name, inner_crop_pct=0.85)
            except Exception as e:
                print("Skip", p, "->", e)

if __name__ == "__main__":
    for cls_dir in sorted([p for p in SRC.iterdir() if p.is_dir()]):
        out_dir = DST / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in cls_dir.iterdir():
            if p.is_file():
                try:
                    sanitize_one(p, out_dir / p.name, inner_crop_pct=0.85)
                except Exception as e:
                    print("Skip", p, "->", e)