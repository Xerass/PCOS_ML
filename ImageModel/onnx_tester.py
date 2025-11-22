#!/usr/bin/env python3
"""Minimal hard-coded ONNX tester for PCOS image model

Behavior (hard-coded):
- ONNX model: ../onnx_models/pcos_image_model.onnx (relative to this script)
- Data dir:  ../data (relative to this script)
- Samples:   10 random images from the data dir
- Preprocessing: exact pipeline from `data_sanitizer.py` (trim, crop, equalize, to_gray3, contain->pad to 384), then resize to 224 and ImageNet normalize.
- Layout: NCHW, dtype=float32
- Output: Applies sigmoid (binary head) if model returns single logit; otherwise uses softmax for multiclass.
- Produces console table and CSV report `onnx_test_report.csv` in the script directory.

"""

from pathlib import Path
import os
import random
import csv
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

import data_sanitizer as ds

HERE = Path(__file__).parent.resolve()
ONNX_PATH = HERE / 'onnx_models' / 'pcos_image_model.onnx'
DATA_DIR = HERE / 'data'
SAMPLE_COUNT = 10

# sanitizer provides IMG_SIZE=384; training pipeline used IMG_SIZE->resize 224 at train time
SANITIZER_SIZE = ds.IMG_SIZE
MODEL_INPUT_SIZE = 224

# ImageNet normalization used in training notebook
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OUT_CSV = HERE / 'onnx_test_report.csv'


def list_images(root: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = [p for p in root.rglob('*') if p.suffix.lower() in exts]
    return files


def apply_sanitizer(img: Image.Image) -> Image.Image:
    """Apply same pipeline as data_sanitizer for a PIL image and return processed PIL image.

    Note: does not read/write disk.
    """
    # convert to RGB (sanitizer expects RGB input)
    img = img.convert('RGB')
    img = ds.trim_black_border(img, threshold=8, margin=4)
    img = ds.center_crop_percent(img, pct=0.85)
    img = ImageOps.equalize(img)
    img = ds.to_gray3(img)
    img = ImageOps.contain(img, (SANITIZER_SIZE, SANITIZER_SIZE))
    bg = Image.new('RGB', (SANITIZER_SIZE, SANITIZER_SIZE), (0, 0, 0))
    bg.paste(img, ((SANITIZER_SIZE - img.size[0]) // 2, (SANITIZER_SIZE - img.size[1]) // 2))
    return bg


def preprocess_for_model(img: Image.Image) -> np.ndarray:
    """Resize to MODEL_INPUT_SIZE, normalize with ImageNet mean/std, convert to NCHW float32."""
    img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    # Apply ImageNet normalization per channel
    arr = (arr - IMAGENET_MEAN.reshape((1, 1, 3))) / IMAGENET_STD.reshape((1, 1, 3))
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    return arr


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def run():
    print('ONNX tester (hard-coded, relative paths)')
    print('Script dir:', HERE)
    print('ONNX:', ONNX_PATH)
    print('Data :', DATA_DIR)

    if not ONNX_PATH.is_file():
        raise FileNotFoundError(f'ONNX model not found at {ONNX_PATH}')
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f'Data dir not found at {DATA_DIR}')

    imgs = list_images(DATA_DIR)
    if not imgs:
        raise RuntimeError('No images found in data dir')

    # make sampling reproducible for demo runs
    random.seed(42)
    sample = random.sample(imgs, min(SAMPLE_COUNT, len(imgs)))
    print('Sampling', len(sample), 'images')

    # load, sanitize, preprocess
    preprocessed = []
    meta = []  # tuples: (path, parent_folder, orig_size)
    for p in sample:
        try:
            im = Image.open(p)
            im.load()
        except Exception as e:
            print('Skip unreadable:', p, e)
            continue

        san = apply_sanitizer(im)
        arr = preprocess_for_model(san)
        preprocessed.append(arr)
        meta.append((str(p), p.parent.name.lower(), im.size))

    if not preprocessed:
        raise RuntimeError('No valid images after preprocessing')

    batch = np.stack(preprocessed, axis=0)  # shape (N, C, H, W)

    # create session and run
    sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: batch})
    out_arr = np.asarray(out[0])

    # Determine class folder order so we know which index corresponds to 'infected'
    class_dirs = sorted([p.name.lower() for p in DATA_DIR.iterdir() if p.is_dir()])
    infected_index = class_dirs.index('infected') if 'infected' in class_dirs else None
    if infected_index is None:
        print('Warning: could not find an "infected" folder under data; defaulting to assume infected=index 0')

    # Interpret outputs: single-logit -> sigmoid; multiclass -> softmax
    N = batch.shape[0]
    results = []
    if out_arr.ndim == 1 and out_arr.shape[0] == N:
        # shape (N,) raw logits for class index 1 (training used ImageFolder labels; sigmoid(logit) ~= P(label==1))
        raw = out_arr.reshape(-1)
        prob_class1 = sigmoid(raw)
        # Convert to probability for 'infected' (positive) depending on folder order used during training
        if infected_index is None or infected_index == 0:
            # if infected is index 0, model's sigmoid gives P(class==1) i.e. P(noninfected)
            prob_pos = 1.0 - prob_class1
        else:
            # infected is index 1, sigmoid already gives P(infected)
            prob_pos = prob_class1
        preds = (prob_pos >= 0.5).astype(int)
        for i in range(N):
            results.append((float(raw[i]), float(prob_pos[i]), int(preds[i])))
    elif out_arr.ndim == 2 and out_arr.shape[0] == N and out_arr.shape[1] == 1:
        raw = out_arr[:, 0]
        prob_class1 = sigmoid(raw)
        if infected_index is None or infected_index == 0:
            prob_pos = 1.0 - prob_class1
        else:
            prob_pos = prob_class1
        preds = (prob_pos >= 0.5).astype(int)
        for i in range(N):
            results.append((float(raw[i]), float(prob_pos[i]), int(preds[i])))
    else:
        # multiclass: pick argmax probability
        if out_arr.ndim == 1:
            out2 = out_arr.reshape(N, -1)
        else:
            out2 = out_arr.reshape(N, -1)
        probs_all = softmax(out2, axis=1)
        preds = np.argmax(probs_all, axis=1)
        probs = probs_all[np.arange(N), preds]
        raw = out2[np.arange(N), preds]
        # For multiclass, decide which class index corresponds to 'infected' and map prediction to infected/not
        for i in range(N):
            pred_class = int(preds[i])
            prob_pred_class = float(probs[i])
            # If the predicted class equals the infected index, mark as infected
            pred_infected = 1 if (infected_index is not None and pred_class == infected_index) else 0
            results.append((float(raw[i]), float(prob_pred_class) if pred_infected else float(1.0 - prob_pred_class), int(pred_infected)))

    # Build report rows and compute simple metrics (assume folder name contains 'infect' for positive)
    rows = []
    y_true = []
    y_pred = []
    for i, ((path, folder, size), (raw, prob, pred)) in enumerate(zip(meta, results)):
        # Robust label extraction:
        # - if folder explicitly contains a negative marker like 'non' treat as negative
        # - if folder explicitly equals or contains 'infect' and does NOT contain a 'non' marker, treat as positive
        # - else mark as unknown (None) and skip from metric computation
        f = folder.lower()
        if ('non' in f) and ('infect' in f):
            true = 0
        elif 'non' in f:
            true = 0
        elif 'infect' in f and ('non' not in f):
            true = 1
        else:
            true = None
        rows.append({'file': path, 'folder': folder, 'true': true, 'pred': pred, 'prob': prob, 'raw': raw, 'size': size})
        y_true.append(true)
        y_pred.append(pred)

    # Write CSV
    with open(OUT_CSV, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'folder', 'true', 'pred', 'prob', 'raw', 'size'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Console table
    print('\nTest report (first {} samples)'.format(len(rows)))
    print(f"{"file":40} {"folder":12} {"true":4} {"pred":4} {"prob":6} {"raw":10} {"size"}")
    print('-' * 100)
    for r in rows:
        print(f"{Path(r['file']).name:40} {r['folder']:12} {r['true']:4} {r['pred']:4} {r['prob']:6.3f} {r['raw']:10.4f} {r['size']}")

    # Simple binary metrics (only if true labels are known)
    known_idx = [i for i, t in enumerate(y_true) if t is not None]
    if known_idx:
        y_true_arr = np.array([y_true[i] for i in known_idx])
        y_pred_arr = np.array([y_pred[i] for i in known_idx])
        acc = (y_true_arr == y_pred_arr).mean()
        tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
        tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
        fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
        fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
        print('\nSummary metrics (only samples with inferred true labels):')
        print(f'  Known samples: {len(known_idx)} / {len(rows)}  Accuracy: {acc:.3f}  TP={tp} FP={fp} TN={tn} FN={fn}')
    else:
        print('\nNo reliable true labels could be inferred from folder names; skipping metrics.')
    print('\nCSV report written to:', OUT_CSV)


if __name__ == '__main__':
    run()
