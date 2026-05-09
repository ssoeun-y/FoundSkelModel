"""
Phase 2: cls-based proposal (identical to Phase 0) + boundary score re-scoring.

Proposal 생성: Phase 0과 완전 동일 (cls_prob > threshold → get_proposal)
Score 재계산:  score = cls_score^(1-alpha) * (start_score[t_s] * end_score[t_e] + 1e-8)^alpha
alpha=0.0    → baseline score (Phase 0과 동일, sanity check용)
alpha>0      → boundary score가 ranking에 기여

Usage:
  python scripts/generate_bbox_phase2.py \
    --src  results/phase1_aux_s2_map_<ts>/detect_each_frame \
    --gt   scripts/V1_Label \
    [--alphas 0.0 0.2 0.4 0.6 0.8] \
    [--threshold 0.02]
"""

import os
import argparse
import numpy as np
NUMBER_LABEL = 52


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_each_frame(path):
    with open(path) as f:
        rows = [l.strip().split(',') for l in f if l.strip()]
    return np.array(rows, dtype=np.float32)   # [T, 54 or 56]


def load_ground(gt_dir, video):
    path = os.path.join(gt_dir, video)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [[float(v) for v in l.replace(',', ' ').split()]
                for l in f if l.strip()]


# ── proposal: identical to Phase 0 get_proposal ───────────────────────────────

def get_proposal(brr):
    arr = np.append(brr, 0)
    proposals, start = [], None
    for i in range(len(arr)):
        if arr[i] == 1:
            if start is None:
                start = i
        elif start is not None:
            proposals.append([start, i])
            start = None
    return proposals


def make_proposals(data, alpha, threshold, snap_k=0, use_loc=False):
    """
    Phase 0과 동일하게 cls threshold → get_proposal로 구간 생성.
    score만 boundary로 재계산 (alpha=0이면 Phase 0 score와 동일).
    snap_k > 0이면 start/end scores argmax로 boundary position refinement.
    use_loc=True이면 cols 56/57 (d_start, d_end) regression으로 boundary 보정.
    """
    cls_prob   = data[:, 2:54]                     # [T, 52]
    has_bnd    = data.shape[1] >= 56
    if has_bnd:
        start_scores = data[:, 54]                 # [T]
        end_scores   = data[:, 55]                 # [T]

    pb_matrix    = cls_prob.T                      # [52, T]
    mask_matrix  = (pb_matrix > threshold).astype(int)
    T_len        = data.shape[0]

    proposals = []
    for cls_idx in range(1, NUMBER_LABEL):         # skip bg (0)
        segments = get_proposal(mask_matrix[cls_idx])

        if has_bnd and snap_k > 0:
            refined = []
            for u, v in segments:
                s_lo = max(0, u - snap_k); s_hi = min(T_len, u + snap_k + 1)
                new_u = s_lo + int(np.argmax(start_scores[s_lo:s_hi]))
                e_lo = max(0, v - snap_k); e_hi = min(T_len, v + snap_k + 1)
                new_v = e_lo + int(np.argmax(end_scores[e_lo:e_hi]))
                if new_u >= new_v:
                    new_u, new_v = u, v
                refined.append([new_u, new_v])
            segments = refined

        if data.shape[1] >= 58 and use_loc:
            d_start_pred = data[:, 56]
            d_end_pred   = data[:, 57]
            refined = []
            for u, v in segments:
                new_u = max(0, int(round(u - d_start_pred[u])))
                new_v = min(T_len - 1, int(round(v + d_end_pred[max(0, v - 1)])))
                if new_u >= new_v:
                    new_u, new_v = u, v
                refined.append([new_u, new_v])
            segments = refined

        for t_s, t_e in segments:
            if t_e <= t_s:
                continue
            cls_score = float(np.mean(cls_prob[t_s:t_e, cls_idx]))

            if has_bnd and alpha > 0:
                bnd_score = float(start_scores[t_s]) * float(end_scores[t_e - 1])
                score = (cls_score ** (1.0 - alpha)) * ((bnd_score + 1e-8) ** alpha)
            else:
                score = cls_score

            proposals.append([cls_idx, t_s, t_e, score])

    return proposals


# ── Soft-NMS (Gaussian decay) ─────────────────────────────────────────────────

def soft_nms(proposals, sigma=0.5, score_thr=1e-4):
    if not proposals:
        return []
    props  = np.array(proposals, dtype=np.float64)
    labels = props[:, 0].astype(int)
    starts = props[:, 1]
    ends   = props[:, 2]
    scores = props[:, 3].copy()

    keep = []
    idxs = list(range(len(props)))
    while idxs:
        best = max(idxs, key=lambda i: scores[i])
        keep.append(best)
        idxs.remove(best)
        for j in idxs:
            if labels[j] != labels[best]:
                continue
            inter = max(0.0, min(ends[best], ends[j]) - max(starts[best], starts[j]))
            union = max(ends[best], ends[j]) - min(starts[best], starts[j])
            iou   = inter / union if union > 0 else 0.0
            scores[j] *= np.exp(-(iou ** 2) / sigma)
        idxs = [j for j in idxs if scores[j] >= score_thr]

    return [[int(props[k, 0]), int(props[k, 1]), int(props[k, 2]), float(scores[k])]
            for k in keep]


# ── mAP helpers ───────────────────────────────────────────────────────────────

def calc_pr(positive, proposal, ground):
    if proposal == 0 or ground == 0:
        return 0.0, 0.0
    return positive / proposal, positive / ground


def t_overlap(prop, gnd):
    l_p, s_p, e_p, _, v_p = prop
    l_g, s_g, e_g, _, v_g = gnd
    if int(l_p) != int(l_g) or v_p != v_g:
        return 0.0
    inter = min(e_p, e_g) - max(s_p, s_g)
    union = max(e_p, e_g) - min(s_p, s_g)
    return inter / union if union > 0 else 0.0


def compute_ap(lst, ratio, ground):
    if not lst or not ground:
        return 0.0
    lst = sorted(lst, key=lambda x: x[3])
    cos_map   = [-1] * len(lst)
    count_map = [0]  * len(ground)
    idx_map   = [[] for _ in range(NUMBER_LABEL)]
    for xi, g in enumerate(ground):
        idx_map[int(g[0])].append(xi)
    for xi, p in enumerate(lst):
        best = ratio
        for yi in idx_map[int(p[0])]:
            ov = t_overlap(p, ground[yi])
            if ov >= best:
                best = ov
                cos_map[xi] = yi
        if cos_map[xi] != -1:
            count_map[cos_map[xi]] += 1
    positive = sum(c > 0 for c in count_map)

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


def compute_map(a_props, a_grounds, theta):
    return sum(compute_ap(a_props[c], theta, a_grounds[c])
               for c in range(1, NUMBER_LABEL)) / (NUMBER_LABEL - 1)


# ── main ──────────────────────────────────────────────────────────────────────

def run_alpha(src_dir, gt_dir, alpha, threshold, use_softnms, snap_k=0, use_loc=False, out_dir=None):
    a_props   = [[] for _ in range(NUMBER_LABEL)]
    a_grounds = [[] for _ in range(NUMBER_LABEL)]

    for video in os.listdir(src_dir):
        data = load_each_frame(os.path.join(src_dir, video))
        gnd  = load_ground(gt_dir, video)
        if not gnd:
            continue

        props = make_proposals(data, alpha, threshold, snap_k=snap_k, use_loc=use_loc)
        if use_softnms:
            props = soft_nms(props)

        if out_dir:
            sub = os.path.join(out_dir, f'alpha{alpha:.1f}', 'detect_result')
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, video), 'w') as ff:
                for lb, st, ed, sc in props:
                    ff.write(f'{int(lb)},{int(st)},{int(ed)},{sc}\n')

        for row in props:
            a_props[int(row[0])].append(row + [video])
        for row in gnd:
            a_grounds[int(row[0])].append(row + [video])

    return a_props, a_grounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',       required=True)
    parser.add_argument('--gt',        required=True)
    parser.add_argument('--alphas',    nargs='+', type=float,
                        default=[0.0, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument('--threshold', type=float, default=0.02,
                        help='cls prob threshold (same as Phase 0)')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument('--softnms',   action='store_true', default=False)
    parser.add_argument('--snap-k',    type=int, default=0,
                        help='boundary snap window half-size (0=disabled)')
    parser.add_argument('--use-loc',   action='store_true', default=False,
                        help='use cols 56/57 (d_start, d_end) for boundary regression correction')
    parser.add_argument('--out',       default=None,
                        help='output directory for detect_result files')
    args = parser.parse_args()

    src_dir = args.src.rstrip('/')
    gt_dir  = args.gt.rstrip('/')

    sample   = os.listdir(src_dir)[0]
    has_bnd  = load_each_frame(os.path.join(src_dir, sample)).shape[1] >= 56
    print(f'\nsrc           : {src_dir}')
    print(f'gt            : {gt_dir}')
    print(f'boundary cols : {has_bnd}')
    print(f'cls threshold : {args.threshold}  (Phase 0 identical)')
    print(f'soft-NMS      : {args.softnms}')
    print(f'snap-k        : {args.snap_k}')
    print(f'use-loc       : {args.use_loc}')
    print(f'alpha sweep   : {args.alphas}')
    if not has_bnd:
        print('[WARN] No boundary score cols (54,55) — all alphas will give same result as alpha=0')
    print('=' * 60)

    header = f"{'alpha':>6} " + ' '.join(f"{'mAP@'+str(t):>9}" for t in args.thresholds)
    print(header)
    print('-' * 60)

    best = {'key': -1, 'alpha': None, 'row': ''}
    for alpha in args.alphas:
        a_props, a_grounds = run_alpha(src_dir, gt_dir, alpha,
                                       args.threshold, args.softnms,
                                       snap_k=args.snap_k, use_loc=args.use_loc,
                                       out_dir=args.out)
        maps = [compute_map(a_props, a_grounds, t) for t in args.thresholds]
        row  = f'{alpha:>6.1f} ' + ' '.join(f'{m:>9.4f}' for m in maps)
        print(row)

        hi_idx = [i for i, t in enumerate(args.thresholds) if t in (0.5, 0.7)]
        key = np.mean([maps[i] for i in hi_idx]) if hi_idx else maps[0]
        if key > best['key']:
            best = {'key': key, 'alpha': alpha, 'row': row}

    print('=' * 60)
    print(f'Best alpha={best["alpha"]:.1f}  (avg mAP@0.5+0.7):')
    print(f'  {best["row"]}')


if __name__ == '__main__':
    main()
