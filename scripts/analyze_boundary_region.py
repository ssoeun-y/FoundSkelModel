"""
Boundary Region vs Interior Analysis
=====================================
GT segment를 3구간으로 분리하여 frame-wise accuracy 측정:
  - start neighborhood  : [gt_start - k, gt_start + k]
  - end neighborhood    : [gt_end   - k, gt_end   + k]
  - interior            : (gt_start + k, gt_end - k)
  - background          : 나머지

사용법:
  python analyze_boundary_region.py \
    --src  results/phase0_best_map_xxx/detect_each_frame \
    --gt   scripts/V1_Label \
    --k    3 \
    --name "Phase 0 baseline"
"""

import os
import argparse
import numpy as np


def load_gt(gt_folder):
    gt = {}
    for fname in os.listdir(gt_folder):
        rows = []
        with open(os.path.join(gt_folder, fname)) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                label = int(float(parts[0]))
                start = int(float(parts[1]))
                end   = int(float(parts[2]))
                rows.append((label, start, end))
        gt[fname] = rows
    return gt


def load_pred(src_folder, fname):
    path = os.path.join(src_folder, fname)
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 54:
                continue
            rows.append([float(x) for x in parts[:54]])
    if not rows:
        return None
    return np.array(rows, dtype=np.float32)


def classify_frames(T, gt_actions, k):
    zones = np.zeros(T, dtype=int)  # 0=background

    for label, s, e in gt_actions:
        if s >= T or e > T:
            continue
        for t in range(s, e):
            if zones[t] == 0:
                zones[t] = 2  # interior
        for t in range(max(0, s - k), min(T, s + k + 1)):
            zones[t] = 1  # start boundary
        for t in range(max(0, e - k), min(T, e + k + 1)):
            zones[t] = 3  # end boundary

    return zones


def analyze(src_folder, gt_folder, k, name):
    gt_all = load_gt(gt_folder)

    stats = {
        'start':      {'correct': 0, 'total': 0},
        'interior':   {'correct': 0, 'total': 0},
        'end':        {'correct': 0, 'total': 0},
        'background': {'correct': 0, 'total': 0},
        'all':        {'correct': 0, 'total': 0},
    }

    zone_map = {0: 'background', 1: 'start', 2: 'interior', 3: 'end'}

    for fname, gt_actions in gt_all.items():
        data = load_pred(src_folder, fname)
        if data is None:
            continue

        T = len(data)
        pred_labels = data[:, 0].astype(int)
        gt_labels   = data[:, 1].astype(int)
        zones       = classify_frames(T, gt_actions, k)

        for t in range(T):
            z = zone_map[zones[t]]
            correct = int(pred_labels[t] == gt_labels[t])
            stats[z]['correct'] += correct
            stats[z]['total']   += 1
            stats['all']['correct'] += correct
            stats['all']['total']   += 1

    print(f"\n{'='*60}")
    print(f"  {name}  (k={k})")
    print(f"{'='*60}")
    print(f"{'Zone':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print(f"{'-'*50}")

    results = {}
    for z in ['start', 'interior', 'end', 'background', 'all']:
        s = stats[z]
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0.0
        results[z] = acc
        marker = ' <-- boundary' if z in ('start', 'end') else ''
        print(f"{z:<15} {acc:>9.2f}% {s['correct']:>10} {s['total']:>10}{marker}")

    boundary_acc = (stats['start']['correct'] + stats['end']['correct']) / \
                   max(stats['start']['total'] + stats['end']['total'], 1) * 100
    interior_acc = stats['interior']['correct'] / max(stats['interior']['total'], 1) * 100
    gap = interior_acc - boundary_acc

    print(f"\n  Boundary accuracy : {boundary_acc:.2f}%")
    print(f"  Interior accuracy : {interior_acc:.2f}%")
    print(f"  Gap (interior - boundary) : {gap:.2f}%  {'[boundary weak]' if gap > 5 else '[balanced]'}")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',  required=True)
    parser.add_argument('--gt',   required=True)
    parser.add_argument('--k',    type=int, default=3)
    parser.add_argument('--name', default='Model')
    args = parser.parse_args()

    analyze(args.src, args.gt, args.k, args.name)
