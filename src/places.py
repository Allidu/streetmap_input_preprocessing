import os
import glob
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image

import torch
from torchvision import transforms
from torchvision.models import resnet18


IN_DIR = "input"              # folder with images
OUT_DIR = "out_scene_scores"   # outputs: csv + contact_sheet
os.makedirs(OUT_DIR, exist_ok=True)

PLACES_DIR = "places365"  # where you store the 2 downloaded files
WEIGHTS_PATH = os.path.join(PLACES_DIR, "resnet18_places365.pth.tar")
CATS_PATH = os.path.join(PLACES_DIR, "categories_places365.txt")

TOPK = 5
DEVICE = "cpu"  # keep CPU for portability; change to "mps" or "cuda" if you know you have it

# contact sheet
THUMB_W = 480
COLS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX


# -------------------------
# Helpers
# -------------------------
def load_categories(path: str) -> List[str]:
    """
    Places365 categories file lines look like:
    '0 /a/airfield  ...'
    or '  0  /a/airfield'
    We'll robustly parse the token that contains '/a/' etc.
    """
    classes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # find the part that contains '/'
            cat = None
            for p in parts:
                if "/" in p:
                    cat = p
                    break
            if cat is None:
                # fallback: last token
                cat = parts[-1]
            # strip leading like '/a/' -> 'airfield'
            cat = cat.replace("\\", "/")
            if cat.startswith("/"):
                cat = cat[1:]
            # remove prefix like 'a/' or 'b/' etc
            if len(cat) >= 2 and cat[1] == "/":
                cat = cat[2:]
            classes.append(cat)
    return classes


def load_places365_model(weights_path: str, device: str = "cpu"):
    model = resnet18(num_classes=365)
    ckpt = torch.load(weights_path, map_location=device)

    # common formats: {'state_dict': ...} or direct state dict
    state_dict = ckpt.get("state_dict", ckpt)

    # remove 'module.' prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=True)
    model.eval()
    model.to(device)

    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return model, tfm


def list_images(in_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.webp", "*.WEBP")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(in_dir, e)))
    return sorted(paths)


def classify_image(model, tfm, classes: List[str], img_path: str, device: str, topk: int) -> List[Tuple[str, float]]:
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        vals, idxs = torch.topk(probs, k=topk)

    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append((classes[i], float(v)))
    return out


def annotate_thumb(bgr: np.ndarray, img_name: str, preds: List[Tuple[str, float]]) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    overlay_h = int(0.30 * h)

    # dark overlay
    cv2.rectangle(out, (0, 0), (w, overlay_h), (0, 0, 0), thickness=-1)
    alpha = 0.55
    out[:overlay_h] = cv2.addWeighted(out[:overlay_h], alpha, bgr[:overlay_h], 1 - alpha, 0)

    y = 24
    cv2.putText(out, img_name, (10, y), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y += 24

    for label, p in preds:
        cv2.putText(out, f"{label}: {p:.2f}", (10, y), FONT, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22

    return out


def make_contact_sheet(thumbs: List[np.ndarray], out_path: str, cols: int, thumb_w: int):
    if not thumbs:
        raise RuntimeError("No thumbnails to render.")

    rows = int(math.ceil(len(thumbs) / cols))
    max_h = max(t.shape[0] for t in thumbs)
    pad = 10

    sheet_w = cols * thumb_w + (cols + 1) * pad
    sheet_h = rows * max_h + (rows + 1) * pad
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    for idx, t in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (thumb_w + pad)
        y0 = pad + r * (max_h + pad)
        hh, ww = t.shape[:2]
        sheet[y0:y0 + hh, x0:x0 + ww] = t

    cv2.imwrite(out_path, sheet)


# -------------------------
# Main
# -------------------------
def main():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_PATH}")
    if not os.path.exists(CATS_PATH):
        raise FileNotFoundError(f"Missing categories: {CATS_PATH}")

    classes = load_categories(CATS_PATH)
    model, tfm = load_places365_model(WEIGHTS_PATH, device=DEVICE)

    paths = list_images(IN_DIR)
    if not paths:
        print(f"No images found in: {IN_DIR}")
        return

    csv_path = os.path.join(OUT_DIR, "scene_scores.csv")
    sheet_path = os.path.join(OUT_DIR, "scene_contact_sheet.jpg")

    thumbs = []

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)

        # header
        header = ["path"]
        for i in range(TOPK):
            header += [f"label_{i+1}", f"prob_{i+1}"]
        w.writerow(header)

        for p in paths:
            try:
                preds = classify_image(model, tfm, classes, p, DEVICE, TOPK)
            except Exception as e:
                print("Failed:", p, "|", e)
                continue

            row = [p]
            for label, prob in preds:
                row += [label, f"{prob:.6f}"]
            w.writerow(row)

            # thumbnail
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            h, ww = bgr.shape[:2]
            scale = THUMB_W / float(ww)
            new_h = int(round(h * scale))
            thumb = cv2.resize(bgr, (THUMB_W, new_h), interpolation=cv2.INTER_AREA)
            thumb = annotate_thumb(thumb, os.path.basename(p), preds)
            thumbs.append(thumb)

            # print quick
            top1 = preds[0]
            print(os.path.basename(p), "->", top1[0], f"{top1[1]:.2f}")

    make_contact_sheet(thumbs, sheet_path, cols=COLS, thumb_w=THUMB_W)

    print("Done.")
    print("CSV:", csv_path)
    print("Sheet:", sheet_path)


if __name__ == "__main__":
    main()
