"""
Visualize start/end boundary scores vs GT boundaries.

Reads detect_each_frame files (cols: pred, gt, cls×52, start_score, end_score).
GT boundaries derived from gt label column (0→nonzero = start, nonzero→0 = end).

Usage:
  python scripts/visualize_boundary.py \
    --src results/xxx/detect_each_frame \
    --out results/xxx/boundary_viz \
    [--n 20] [--seed 0]
"""

import os
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── helpers ───────────────────────────────────────────────────────────────────

def load_file(path):
    with open(path) as f:
        rows = [l.strip().split(',') for l in f if l.strip()]
    data = np.array(rows, dtype=np.float32)   # [T, 2+52+2] or [T, 2+52]
    return data


def get_gt_boundaries(gt_labels):
    """Returns (start_frames, end_frames) from integer label sequence."""
    T = len(gt_labels)
    starts, ends = [], []
    for t in range(1, T):
        if gt_labels[t] != 0 and gt_labels[t - 1] == 0:
            starts.append(t)
        if gt_labels[t] == 0 and gt_labels[t - 1] != 0:
            ends.append(t - 1)
    return starts, ends


def boundary_recall(scores, gt_pos, threshold, delta):
    """Fraction of GT positions where peak within ±delta exceeds threshold."""
    if not gt_pos:
        return float('nan')
    T = len(scores)
    hit = 0
    for t in gt_pos:
        lo, hi = max(0, t - delta), min(T, t + delta + 1)
        if scores[lo:hi].max() >= threshold:
            hit += 1
    return hit / len(gt_pos)


def avg_score_at_boundary(scores, gt_pos):
    if not gt_pos:
        return float('nan')
    return float(np.mean([scores[t] for t in gt_pos if 0 <= t < len(scores)]))


# ── per-file plot ─────────────────────────────────────────────────────────────

def plot_file(data, filename, out_dir):
    has_boundary_scores = data.shape[1] >= 56

    T          = data.shape[0]
    gt_labels  = data[:, 1].astype(int)
    cls_max    = data[:, 2:54].max(axis=1)   # max cls prob over classes [T]

    frames = np.arange(T)
    gt_starts, gt_ends = get_gt_boundaries(gt_labels)

    fig, axes = plt.subplots(3 if has_boundary_scores else 2, 1,
                             figsize=(14, 7 if has_boundary_scores else 5),
                             sharex=True)

    # ── (1) GT action label timeline ──────────────────────────────────────────
    ax = axes[0]
    action_mask = (gt_labels != 0).astype(float)
    ax.fill_between(frames, 0, action_mask, alpha=0.3, color='steelblue', label='GT action')
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel('GT label')
    for t in gt_starts:
        ax.axvline(t, color='green', lw=1.5, linestyle='--', alpha=0.8)
    for t in gt_ends:
        ax.axvline(t, color='red', lw=1.5, linestyle=':', alpha=0.8)
    ax.legend(handles=[
        mpatches.Patch(color='steelblue', alpha=0.3, label='GT action'),
        plt.Line2D([0], [0], color='green', ls='--', label='GT start'),
        plt.Line2D([0], [0], color='red',   ls=':',  label='GT end'),
    ], fontsize=7, loc='upper right')
    ax.set_title(filename, fontsize=9)

    # ── (2) cls max prob ──────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(frames, cls_max, color='royalblue', lw=1, label='cls max prob')
    for t in gt_starts:
        ax.axvline(t, color='green', lw=1.2, linestyle='--', alpha=0.6)
    for t in gt_ends:
        ax.axvline(t, color='red', lw=1.2, linestyle=':', alpha=0.6)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel('cls prob')
    ax.legend(fontsize=7, loc='upper right')

    # ── (3) start / end scores (only if cols 54,55 exist) ────────────────────
    if has_boundary_scores:
        start_scores = data[:, 54]
        end_scores   = data[:, 55]

        ax = axes[2]
        ax.plot(frames, start_scores, color='green', lw=1.2, label='start score')
        ax.plot(frames, end_scores,   color='red',   lw=1.2, label='end score', linestyle='--')
        for t in gt_starts:
            ax.axvline(t, color='green', lw=1.5, linestyle='--', alpha=0.7)
        for t in gt_ends:
            ax.axvline(t, color='red', lw=1.5, linestyle=':', alpha=0.7)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel('boundary score')
        ax.set_xlabel('frame')
        ax.legend(fontsize=7, loc='upper right')

        # metrics annotation
        s_rec = boundary_recall(start_scores, gt_starts, threshold=0.3, delta=2)
        e_rec = boundary_recall(end_scores,   gt_ends,   threshold=0.3, delta=2)
        s_avg = avg_score_at_boundary(start_scores, gt_starts)
        e_avg = avg_score_at_boundary(end_scores,   gt_ends)
        ax.text(0.01, 0.92,
                f'start recall@0.3,δ2={s_rec:.2f}  avg@GT={s_avg:.3f}\n'
                f'end   recall@0.3,δ2={e_rec:.2f}  avg@GT={e_avg:.3f}',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    else:
        axes[-1].set_xlabel('frame')

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename + '.png')
    plt.savefig(out_path, dpi=100)
    plt.close()
    return out_path


# ── aggregate metrics ─────────────────────────────────────────────────────────

def compute_global_metrics(src_dir):
    files = os.listdir(src_dir)
    metrics = {'s_avg': [], 'e_avg': [],
               'rec_s_d1': [], 'rec_s_d2': [], 'rec_s_d3': [],
               'rec_e_d1': [], 'rec_e_d2': [], 'rec_e_d3': []}

    has_scores = None
    for fname in files:
        data = load_file(os.path.join(src_dir, fname))
        if has_scores is None:
            has_scores = data.shape[1] >= 56
        if not has_scores:
            break

        gt_labels = data[:, 1].astype(int)
        gt_starts, gt_ends = get_gt_boundaries(gt_labels)
        start_scores = data[:, 54]
        end_scores   = data[:, 55]

        if gt_starts:
            metrics['s_avg'].append(avg_score_at_boundary(start_scores, gt_starts))
            for d, key in [(1, 'rec_s_d1'), (2, 'rec_s_d2'), (3, 'rec_s_d3')]:
                v = boundary_recall(start_scores, gt_starts, 0.3, d)
                if not np.isnan(v):
                    metrics[key].append(v)
        if gt_ends:
            metrics['e_avg'].append(avg_score_at_boundary(end_scores, gt_ends))
            for d, key in [(1, 'rec_e_d1'), (2, 'rec_e_d2'), (3, 'rec_e_d3')]:
                v = boundary_recall(end_scores, gt_ends, 0.3, d)
                if not np.isnan(v):
                    metrics[key].append(v)

    if not has_scores:
        print('[INFO] No boundary score columns (54,55) found — skipping metrics.')
        return

    print('\n[Global Boundary Metrics]  (threshold=0.3)')
    print(f'  avg start score @ GT start : {np.mean(metrics["s_avg"]):.4f}')
    print(f'  avg end   score @ GT end   : {np.mean(metrics["e_avg"]):.4f}')
    print(f'  start recall @ δ=1,2,3     : '
          f'{np.mean(metrics["rec_s_d1"]):.3f} / '
          f'{np.mean(metrics["rec_s_d2"]):.3f} / '
          f'{np.mean(metrics["rec_s_d3"]):.3f}')
    print(f'  end   recall @ δ=1,2,3     : '
          f'{np.mean(metrics["rec_e_d1"]):.3f} / '
          f'{np.mean(metrics["rec_e_d2"]):.3f} / '
          f'{np.mean(metrics["rec_e_d3"]):.3f}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True,
                        help='path to detect_each_frame directory')
    parser.add_argument('--out', default=None,
                        help='output directory for plots (default: src/../boundary_viz)')
    parser.add_argument('--n', default=20, type=int,
                        help='number of files to plot (0 = all)')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    src_dir = args.src.rstrip('/')
    out_dir = args.out or os.path.join(os.path.dirname(src_dir), 'boundary_viz')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(os.listdir(src_dir))
    if args.n > 0 and args.n < len(files):
        random.seed(args.seed)
        files = random.sample(files, args.n)

    print(f'Plotting {len(files)} files → {out_dir}')
    for fname in files:
        data = load_file(os.path.join(src_dir, fname))
        plot_file(data, fname, out_dir)

    compute_global_metrics(src_dir)
    print(f'\nDone. Plots saved to {out_dir}/')


if __name__ == '__main__':
    main()
