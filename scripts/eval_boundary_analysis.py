"""
eval_boundary_analysis.py
─────────────────────────
GT instance의 boundary region(±k frames)에서
prediction이 얼마나 정확한지 분석합니다.

지표:
  - Boundary precision: boundary 구간에서 prediction이 GT와 얼마나 겹치는지
  - Interior precision: interior 구간에서의 같은 지표
  - Start/End error: 예측 start/end와 GT start/end의 평균 오차 (프레임)

사용법:
  python scripts/eval_boundary_analysis.py \
    --results results/phase0_best_map_2026_04_20_181201/detect_result \
              results/causal_dste_v3_aux_eval_2026_05_05_195220/detect_result \
    --labels  "DSTE (Offline)" "CausalDSTEAux (Online)" \
    --gt      scripts/V1_Label \
    --k       10
"""

import argparse
import os
import numpy as np
from pathlib import Path


def load_gt(path):
    if not os.path.exists(path):
        return []
    instances = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            cls   = int(float(parts[0]))
            start = int(float(parts[1]))
            end   = int(float(parts[2]))
            instances.append((cls, start, end))
    return instances


def load_pred(path):
    if not os.path.exists(path):
        return []
    preds = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            cls   = int(float(parts[0]))
            start = int(float(parts[1]))
            end   = int(float(parts[2]))
            score = float(parts[3])
            preds.append((cls, start, end, score))
    return preds


def compute_iou(s1, e1, s2, e2):
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def analyze(result_dir, gt_dir, k=10, iou_thr=0.5):
    """
    각 GT instance에 대해 가장 잘 매칭되는 prediction을 찾고
    boundary / interior 오차를 분석합니다.
    """
    start_errors_boundary = []
    start_errors_interior = []
    end_errors_boundary   = []
    end_errors_interior   = []
    matched_boundary = 0
    missed_boundary  = 0
    matched_interior = 0
    missed_interior  = 0

    videos = sorted(os.listdir(result_dir))

    for video in videos:
        gt_path   = os.path.join(gt_dir,     video)
        pred_path = os.path.join(result_dir, video)

        gt_instances   = load_gt(gt_path)
        pred_instances = load_pred(pred_path)

        for gt_cls, gt_s, gt_e in gt_instances:
            duration = gt_e - gt_s
            if duration <= 0:
                continue

            # boundary region: 시작 ±k, 끝 ±k
            # interior: boundary 제외한 중간 구간
            interior_s = gt_s + k
            interior_e = gt_e - k
            has_interior = interior_s < interior_e

            # 같은 class의 prediction 중 IoU 최대인 것 찾기
            best_iou  = 0.0
            best_pred = None
            for pred_cls, pred_s, pred_e, score in pred_instances:
                if pred_cls != gt_cls:
                    continue
                iou = compute_iou(pred_s, pred_e, gt_s, gt_e)
                if iou > best_iou:
                    best_iou  = iou
                    best_pred = (pred_s, pred_e)

            # Boundary 매칭 분석
            if best_iou >= iou_thr and best_pred is not None:
                pred_s, pred_e = best_pred
                start_err = abs(pred_s - gt_s)
                end_err   = abs(pred_e - gt_e)

                # start error → boundary region
                start_errors_boundary.append(start_err)
                end_errors_boundary.append(end_err)
                matched_boundary += 1

                # interior: prediction이 interior 구간을 얼마나 커버하는지
                if has_interior:
                    inter_cover = max(0, min(pred_e, interior_e) - max(pred_s, interior_s))
                    inter_total = interior_e - interior_s
                    if inter_total > 0:
                        matched_interior += 1
                        # interior error: interior 구간에서의 미커버 비율
                        missed_frac = 1.0 - inter_cover / inter_total
                        start_errors_interior.append(missed_frac * duration)
            else:
                missed_boundary += 1
                if has_interior:
                    missed_interior += 1

    total = matched_boundary + missed_boundary
    results = {
        "total_gt":              total,
        "boundary_recall":       matched_boundary / total if total > 0 else 0.0,
        "mean_start_err":        np.mean(start_errors_boundary) if start_errors_boundary else 0.0,
        "mean_end_err":          np.mean(end_errors_boundary)   if end_errors_boundary   else 0.0,
        "mean_boundary_err":     np.mean(start_errors_boundary + end_errors_boundary) if start_errors_boundary else 0.0,
        "matched":               matched_boundary,
        "missed":                missed_boundary,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+', required=True,
                        help='detect_result 폴더들')
    parser.add_argument('--labels',  nargs='+', required=True,
                        help='각 폴더의 레이블')
    parser.add_argument('--gt',      required=True)
    parser.add_argument('--k',       type=int, default=10,
                        help='boundary region 크기 (±k frames)')
    parser.add_argument('--iou-thr', type=float, default=0.5)
    args = parser.parse_args()

    assert len(args.results) == len(args.labels)

    print(f"\nboundary_k = ±{args.k} frames,  IoU threshold = {args.iou_thr}")
    print("=" * 65)
    print(f"{'Model':<30} {'Recall':>8} {'Start err':>10} {'End err':>10} {'Bnd err':>10}")
    print("-" * 65)

    for result_dir, label in zip(args.results, args.labels):
        r = analyze(result_dir, args.gt, k=args.k, iou_thr=args.iou_thr)
        print(f"{label:<30} "
              f"{r['boundary_recall']*100:>7.1f}% "
              f"{r['mean_start_err']:>10.1f} "
              f"{r['mean_end_err']:>10.1f} "
              f"{r['mean_boundary_err']:>10.1f}")

    print("=" * 65)
    print(f"(Start/End err: 프레임 단위 평균 오차, Bnd err: 둘의 평균)")


if __name__ == '__main__':
    main()
