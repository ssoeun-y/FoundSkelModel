"""
t-IoU threshold sweep: diagnose classification error vs localization error.

  If mAP@0.1 >> mAP@0.5  → localization (boundary) is the bottleneck
  If mAP@0.1 ≈  mAP@0.5  → classification is the bottleneck

Usage:
  SOURCE_FOLDER=./results/xxx/detect_result/ \
  GROUND_FOLDER=./scripts/V1_Label/ \
  python scripts/analyze_tiou.py [--thresholds 0.1 0.3 0.5 0.7] [--tag MyModel]
"""

import os
import sys
import argparse
import numpy as np

number_label = 52

# ── core helpers (identical logic to cal_mAP.py) ──────────────────────────────

def calc_pr(positive, proposal, ground):
    if proposal == 0 or ground == 0:
        return 0.0, 0.0
    return positive / proposal, positive / ground


def overlap(prop, gnd):
    l_p, s_p, e_p, _, v_p = prop
    l_g, s_g, e_g, _, v_g = gnd
    if int(l_p) != int(l_g) or v_p != v_g:
        return 0.0
    inter = min(e_p, e_g) - max(s_p, s_g)
    union = max(e_p, e_g) - min(s_p, s_g)
    return inter / union if union > 0 else 0.0


def match(lst, ratio, ground):
    cos_map = [-1] * len(lst)
    count_map = [0] * len(ground)
    index_map = [[] for _ in range(number_label)]
    for xi, g in enumerate(ground):
        index_map[int(g[0])].append(xi)

    for xi, p in enumerate(lst):
        best_ov = ratio  # must exceed threshold
        for yi in index_map[int(p[0])]:
            ov = overlap(p, ground[yi])
            if ov >= best_ov:
                best_ov = ov
                cos_map[xi] = yi
        if cos_map[xi] != -1:
            count_map[cos_map[xi]] += 1

    positive = sum(c > 0 for c in count_map)
    return cos_map, count_map, positive


def ap(lst, ratio, ground):
    if not lst or not ground:
        return 0.0
    lst = sorted(lst, key=lambda x: x[3])  # ascending confidence
    cos_map, count_map, positive = match(lst, ratio, ground)
    score = 0.0
    n_prop = len(lst)
    n_gnd  = len(ground)
    old_p, old_r = calc_pr(positive, n_prop, n_gnd)

    for x in range(len(lst)):
        n_prop -= 1
        if cos_map[x] == -1:
            continue
        count_map[cos_map[x]] -= 1
        if count_map[cos_map[x]] == 0:
            positive -= 1
        p, r = calc_pr(positive, n_prop, n_gnd)
        if p > old_p:
            old_p = p
        score += old_p * (old_r - r)
        old_r = r
    return score


# ── load data ─────────────────────────────────────────────────────────────────

def load_data(source_folder, ground_folder):
    a_props   = [[] for _ in range(number_label)]
    a_grounds = [[] for _ in range(number_label)]
    all_props   = []
    all_grounds = []

    for video in os.listdir(source_folder):
        gt_path = os.path.join(ground_folder, video)
        if not os.path.exists(gt_path):
            continue
        with open(os.path.join(source_folder, video)) as f:
            prop = [[float(v) for v in l.replace(',', ' ').split()] for l in f if l.strip()]
        with open(gt_path) as f:
            gnd  = [[float(v) for v in l.replace(',', ' ').split()] for l in f if l.strip()]

        for row in prop:
            row.append(video)
            a_props[int(row[0])].append(row)
            all_props.append(row)
        for row in gnd:
            row.append(video)
            a_grounds[int(row[0])].append(row)
            all_grounds.append(row)

    return a_props, a_grounds, all_props, all_grounds


# ── per-class localization analysis ───────────────────────────────────────────

def localization_gap(a_props, a_grounds, lo=0.1, hi=0.5):
    """Per-class mAP gap between lo and hi IoU thresholds."""
    gaps = {}
    for c in range(1, number_label):
        ap_lo = ap(a_props[c], lo, a_grounds[c])
        ap_hi = ap(a_props[c], hi, a_grounds[c])
        if ap_lo > 0:
            gaps[c] = (ap_lo, ap_hi, ap_lo - ap_hi)
    return gaps


# ── boundary precision: average IoU of matched proposals ──────────────────────

def avg_iou_of_matches(all_props, all_grounds, ratio=0.1):
    """Among proposals that match a GT at ratio, what is their actual avg IoU?"""
    lst = sorted(all_props, key=lambda x: x[3])
    cos_map, _, _ = match(lst, ratio, all_grounds)
    ious = []
    for xi, yi in enumerate(cos_map):
        if yi != -1:
            ious.append(overlap(lst[xi], all_grounds[yi]))
    return float(np.mean(ious)) if ious else 0.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    source_folder = os.environ.get('SOURCE_FOLDER', './detect_result/')
    ground_folder = os.environ.get('GROUND_FOLDER', './scripts/V1_Label/')
    if not source_folder.endswith('/'):
        source_folder += '/'
    if not ground_folder.endswith('/'):
        ground_folder += '/'

    print(f'\nsrc : {source_folder}')
    print(f'gt  : {ground_folder}')
    tag_str = f'  [{args.tag}]' if args.tag else ''
    print(f'{"=" * 52}{tag_str}')

    a_props, a_grounds, all_props, all_grounds = load_data(source_folder, ground_folder)
    n_gt    = len(all_grounds)
    n_prop  = len(all_props)
    print(f'proposals : {n_prop}   ground-truth : {n_gt}')
    print(f'{"─" * 52}')

    header = f"{'t-IoU':>8}  {'mAP_action':>12}  {'AP_all':>10}  {'F1@thr':>8}"
    print(header)
    print(f'{"─" * 52}')

    results = {}
    for theta in args.thresholds:
        map_action = sum(
            ap(a_props[c], theta, a_grounds[c]) for c in range(1, number_label)
        ) / (number_label - 1)

        ap_all = ap(all_props, theta, all_grounds)

        cos_map, count_map, positive = match(all_props, theta, all_grounds)
        prec, rec = calc_pr(positive, n_prop, n_gt)
        denom = prec + rec
        f1_score = 2 * prec * rec / denom if denom > 0 else 0.0

        results[theta] = dict(map_action=map_action, ap_all=ap_all, f1=f1_score)
        print(f'{theta:>8.2f}  {map_action:>12.4f}  {ap_all:>10.4f}  {f1_score:>8.4f}')

    print(f'{"─" * 52}')

    # ── diagnosis ─────────────────────────────────────────────────────────────
    thrs = sorted(args.thresholds)
    if len(thrs) >= 2:
        lo, hi = thrs[0], thrs[-1]
        map_lo = results[lo]['map_action']
        map_hi = results[hi]['map_action']
        drop   = map_lo - map_hi
        rel    = drop / map_lo if map_lo > 0 else 0

        print(f'\n[Diagnosis]')
        print(f'  mAP@{lo:.1f} = {map_lo:.4f}   mAP@{hi:.1f} = {map_hi:.4f}')
        print(f'  Drop = {drop:.4f}  ({rel*100:.1f}% relative)')
        if rel > 0.35:
            verdict = 'LOCALIZATION (boundary) is the primary bottleneck'
        elif rel < 0.15:
            verdict = 'CLASSIFICATION is the primary bottleneck'
        else:
            verdict = 'Mixed: both boundary error and classification error'
        print(f'  → {verdict}')

        avg_iou = avg_iou_of_matches(all_props, all_grounds, ratio=lo)
        print(f'  Avg IoU of matched proposals (IoU≥{lo:.1f}): {avg_iou:.4f}')

    # ── top classes with largest localization gap ──────────────────────────────
    if len(thrs) >= 2:
        lo2, hi2 = thrs[0], thrs[1] if len(thrs) == 2 else 0.5
        gaps = localization_gap(a_props, a_grounds, lo=lo2, hi=hi2)
        sorted_gaps = sorted(gaps.items(), key=lambda x: -x[1][2])[:10]
        print(f'\n[Top-10 classes by localization gap  AP@{lo2:.1f} → AP@{hi2:.1f}]')
        print(f'  {"cls":>4}  {"AP@lo":>8}  {"AP@hi":>8}  {"gap":>8}')
        for cls, (a_lo, a_hi, gap) in sorted_gaps:
            print(f'  {cls:>4}  {a_lo:>8.4f}  {a_hi:>8.4f}  {gap:>8.4f}')

    print('=' * 52)


if __name__ == '__main__':
    main()
