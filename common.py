"""
label parsing and the plucking-point rule.

"""
from pathlib import Path
import numpy as np

NK = 6
PICK_RULE = ("lerp", 4, 5, 0.5)  
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def picking_point(kpts, rule=PICK_RULE, min_conf=0.5):
    k = np.asarray(kpts, dtype=float)
    if k.shape[0] == 1:  
        rule = ("kpt", 0)
    conf = k[:, 2] if k.shape[1] == 3 else np.ones(len(k))
    if rule[0] == "kpt":
        i = rule[1]
        return (float(k[i, 0]), float(k[i, 1])) if conf[i] >= min_conf else None
    _, i, j, t = rule
    if min(conf[i], conf[j]) < min_conf:
        return None
    p = (1 - t) * k[i, :2] + t * k[j, :2]
    return float(p[0]), float(p[1])


def parse_raw_line(line, img_w, img_h, nk=NK):
    v = [float(t) for t in line.split()]
    n = len(v)
    if n == 5:  
        kp = np.zeros((nk, 3))
    elif n == 5 + 2 * nk:
        pts = np.array(v[5:]).reshape(nk, 2)
        vis = np.where((pts[:, 0] < 0) & (pts[:, 1] < 0), 0.0, 2.0)
        kp = np.column_stack([pts, vis])
    elif n == 5 + 3 * nk:
        kp = np.array(v[5:]).reshape(nk, 3)
    else:
        raise ValueError(f"{n} columns; expected 5, {5 + 2 * nk} or {5 + 3 * nk} for {nk} keypoints")
    box = np.array(v[1:5])
    if max(box.max(), kp[:, :2].max()) > 1.5:  
        box /= [img_w, img_h, img_w, img_h]
        kp[:, 0] /= img_w
        kp[:, 1] /= img_h
    kp[:, :2] = np.clip(kp[:, :2], 0.0, 1.0)
    kp[kp[:, 2] == 0, :2] = 0.0
    return int(v[0]), box, kp


def find_pairs(root):
    root = Path(root)
    labels = {p.stem: p for p in root.rglob("*.txt") if p.stem != "classes"}
    pairs = [(p, labels[p.stem]) for p in sorted(root.rglob("*"))
             if p.suffix.lower() in IMG_EXTS and p.stem in labels]
    return pairs