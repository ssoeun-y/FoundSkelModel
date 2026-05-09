

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

N_ACTION_CLS = 51   # class 1~51 (class 0 = 배경, 제외)


def load_pred(path: Path):
    """col2~53 반환: shape (T, 52), index 0=class0(배경), 1=class1, ..., 51=class51"""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 2:54]   # (T, 52), 0-indexed


def load_gt(path: Path):
    """GT: class_id(1-indexed), start, end 반환"""
    if not path.exists():
        return []
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            cls   = int(float(parts[0]))   # 1-indexed, 변환 없이 그대로
            start = int(float(parts[1]))
            end   = int(float(parts[2]))
            instances.append((cls, start, end))
    return instances


def build_frame_labels(instances, T: int):
    """
    (T, 52) GT 행렬
    GT class c (1-indexed) → labels[:, c] = 1
    배경(class 0) 열은 쓰지 않음 (AP 계산 시 class 1~51만 사용)
    """
    labels = np.zeros((T, 52), dtype=np.float32)
    for cls, start, end in instances:
        if cls < 1 or cls > 51:
            continue
        labels[max(0, start): min(T, end + 1), cls] = 1.0
    return labels


def compute_perframe_map(src_dir: Path, gt_dir: Path):
    # class 1~51 (0-indexed col 1~51 in scores)
    all_scores = [[] for _ in range(52)]
    all_labels = [[] for _ in range(52)]

    pred_files = sorted(src_dir.glob("*.txt"))
    print(f"총 비디오 수: {len(pred_files)}")

    for pred_path in pred_files:
        vid       = pred_path.stem
        scores    = load_pred(pred_path)       # (T, 52), col index = class index (0-indexed)
        T         = scores.shape[0]
        instances = load_gt(gt_dir / f"{vid}.txt")
        labels    = build_frame_labels(instances, T)  # (T, 52)

        for c in range(1, 52):   # class 1~51만 (0=배경 제외)
            all_scores[c].append(scores[:, c])
            all_labels[c].append(labels[:, c])

    per_class_ap = np.full(52, np.nan)
    for c in range(1, 52):
        y_score = np.concatenate(all_scores[c])
        y_true  = np.concatenate(all_labels[c])
        if y_true.sum() == 0:
            continue
        per_class_ap[c] = average_precision_score(y_true, y_score)

    valid = ~np.isnan(per_class_ap)
    mAP   = per_class_ap[valid].mean()
    print(f"유효 클래스 수: {valid.sum()} / 51")
    return per_class_ap, mAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",  required=True)
    parser.add_argument("--gt",   required=True)
    parser.add_argument("--out",  default=None)
    args = parser.parse_args()

    per_class_ap, mAP = compute_perframe_map(Path(args.src), Path(args.gt))

    print(f"\n{'='*40}")
    print(f"Per-frame mAP = {mAP*100:.2f}%")
    print(f"{'='*40}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "mAP": float(mAP),
                "per_class_ap": per_class_ap.tolist(),
                "src": str(args.src),
            }, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
