"""
PKU-MMD style evaluation: mAPa (action-level) and mAPv (video-level)
at overlap ratio theta=0.5

GT format  : class,start,end,confidence  (e.g. 28,271,389,2)
Pred format: class,start,end,score       (e.g. 1,4866,4868,0.022)

Usage:
    python eval_pku_mmd.py \
        --pred results/phase3_best_eval_.../detect_result/ \
        --gt   ./scripts/V1_Label/ \
        --theta 0.5
"""

import os, argparse
import numpy as np
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred',  required=True, help='detect_result folder')
    p.add_argument('--gt',    required=True, help='GT label folder')
    p.add_argument('--theta', type=float, default=0.5, help='IoU threshold')
    return p.parse_args()

# ── I/O ──────────────────────────────────────────────────────────────────────

def load_gt(gt_dir):
    """Returns dict: filename → list of (cls, start, end)"""
    gt = {}
    for fname in os.listdir(gt_dir):
        rows = []
        with open(os.path.join(gt_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                cls, start, end = int(parts[0]), int(parts[1]), int(parts[2])
                rows.append((cls, start, end))
        gt[fname] = rows
    return gt

def load_pred(pred_dir):
    """Returns dict: filename → list of (cls, start, end, score)"""
    pred = {}
    for fname in os.listdir(pred_dir):
        rows = []
        with open(os.path.join(pred_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                cls, start, end, score = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
                rows.append((cls, start, end, score))
        pred[fname] = rows
    return pred

# ── IoU ──────────────────────────────────────────────────────────────────────

def iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0

# ── AP (standard interpolated 11-point) ──────────────────────────────────────

def compute_ap(scores, tp_flags, n_gt):
    """
    scores   : list of floats (descending sorted externally)
    tp_flags : list of 0/1 matching scores
    n_gt     : total number of GT instances for this group
    """
    if n_gt == 0:
        return 0.0
    tp = np.array(tp_flags, dtype=np.float32)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)
    recall    = cum_tp / n_gt

    # interpolated AP
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p = precision[recall >= thr]
        ap += (p.max() if len(p) > 0 else 0.0)
    return ap / 11.0

# ── mAPa: per-action-class AP, then mean ─────────────────────────────────────

def compute_mAPa(gt_all, pred_all, theta):
    """
    For each action class c:
      - collect all predictions of class c across all videos (sorted by score desc)
      - greedily match to GT of class c (one GT can be matched only once)
      - compute AP
    Return mean over all classes that have at least one GT instance.
    """
    all_classes = set()
    for rows in gt_all.values():
        for cls, s, e in rows:
            all_classes.add(cls)

    ap_per_class = {}
    for c in sorted(all_classes):
        preds_c = []
        for fname, rows in pred_all.items():
            for cls, s, e, score in rows:
                if cls == c:
                    preds_c.append((score, fname, s, e))
        preds_c.sort(key=lambda x: -x[0])

        gt_c = defaultdict(list)
        n_gt = 0
        for fname, rows in gt_all.items():
            for cls, s, e in rows:
                if cls == c:
                    gt_c[fname].append([s, e, False])
                    n_gt += 1

        tp_flags = []
        for score, fname, ps, pe in preds_c:
            matched = False
            for g in gt_c[fname]:
                if g[2]:
                    continue
                if iou(ps, pe, g[0], g[1]) >= theta:
                    g[2] = True
                    matched = True
                    break
            tp_flags.append(1 if matched else 0)

        ap_per_class[c] = compute_ap(
            [x[0] for x in preds_c], tp_flags, n_gt)

    mAPa = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    return mAPa, ap_per_class

# ── mAPv: per-video AP, then mean ────────────────────────────────────────────

def compute_mAPv(gt_all, pred_all, theta):
    """
    For each video:
      - sort all predictions by score desc
      - greedily match to GT (class must match)
      - compute AP
    Return mean over all videos.
    """
    ap_per_video = {}
    for fname in gt_all:
        gt_v   = [[cls, s, e, False] for cls, s, e in gt_all[fname]]
        pred_v = sorted(pred_all.get(fname, []), key=lambda x: -x[3])
        n_gt   = len(gt_v)

        tp_flags = []
        for cls, ps, pe, score in pred_v:
            matched = False
            for g in gt_v:
                if g[3]:
                    continue
                if g[0] == cls and iou(ps, pe, g[1], g[2]) >= theta:
                    g[3] = True
                    matched = True
                    break
            tp_flags.append(1 if matched else 0)

        scores = [x[3] for x in pred_v]
        ap_per_video[fname] = compute_ap(scores, tp_flags, n_gt)

    mAPv = np.mean(list(ap_per_video.values())) if ap_per_video else 0.0
    return mAPv, ap_per_video

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Loading GT   : {args.gt}")
    print(f"Loading Pred : {args.pred}")
    print(f"IoU threshold: {args.theta}")

    gt_all   = load_gt(args.gt)
    pred_all = load_pred(args.pred)

    print(f"GT  files: {len(gt_all)}")
    print(f"Pred files: {len(pred_all)}")
    missing = set(gt_all.keys()) - set(pred_all.keys())
    if missing:
        print(f"[WARN] {len(missing)} GT files have no predictions — treated as AP=0 for mAPv")

    mAPa, ap_per_class = compute_mAPa(gt_all, pred_all, args.theta)
    mAPv, ap_per_video = compute_mAPv(gt_all, pred_all, args.theta)

    print("\n" + "="*50)
    print(f"  mAPa @ theta={args.theta:.1f} : {mAPa*100:.2f}%")
    print(f"  mAPv @ theta={args.theta:.1f} : {mAPv*100:.2f}%")
    print("="*50)

    print("\nPer-class AP (mAPa breakdown):")
    for c, ap in sorted(ap_per_class.items()):
        print(f"  class {c:3d}: {ap*100:.2f}%")

if __name__ == '__main__':
    main()
