# Boundary-Aware Online Skeleton Action Detection

> **연구 주제:** Skeleton 기반 Online(Causal) Action Detection에서 boundary localization 문제 해결
> **Backbone:** DSTE — [wengwanjiang/FoundSkelModel](https://github.com/wengwanjiang/FoundSkelModel) fork
> **Dataset:** PKU-MMD v1 cross-subject
> **Evaluation:** mAP@IoU (0.1, 0.3, 0.5, 0.7)

---

## 데이터 및 체크포인트 설정

원본 저자 제공 OneDrive:
https://onedrive.live.com/?id=70B7AF2AB29B610A%21s3a3c4ef3046c47d0a179a9da0c2eb7d4

다운로드 후 아래 구조로 배치:
FoundSkelModel/
├── checkpoint/
│   ├── ntu60_xs_j_dste/
│   │   └── ntu60_xs_joint_dste.pth.tar
│   └── pku_v2_xs_joint_dste.pth.tar
├── data/
│   ├── PKUv1_xsub_train.pkl
│   └── PKUv1_xsub_val.pkl
└── scripts/
└── V1_Label/

> data/, checkpoint/, results/ 는 .gitignore에 포함되어 git에 올라가지 않음

---

## 공정성 원칙 (전 실험 고정)

| 항목 | 값 |
|------|----|
| Backbone | Vanilla DSTE |
| Pretrained | checkpoint/ntu60_xs_j_dste/ntu60_xs_joint_dste.pth.tar |
| Dataset | PKU-MMD v1 cross-subject |
| Epochs / Batch / LR | 150 / 256 / 0.005 (SGD) |
| LR schedule | epoch 120 x0.1, epoch 140 x0.1 |
| Evaluation | scripts/generate_bbox_phase2.py |

---

## 추가된 파일 설명

### model/

| 파일 | 설명 |
|------|------|
| DSTE_causal.py | CausalTemporalLinear (lower-triangular mask), DSTECausal (causal attention mask). pretrained weight 완전 호환 |
| DSTE_causal_aux.py | DownstreamCausalAux: DSTECausal + start/end boundary head |
| AFCM.py | Anticipatory Future Context Module: 과거 feature로 미래 K프레임 예측 |
| DSTE_causal_afcm.py | DownstreamCausalAFCM: CausalDSTEAux + AFCM + offline teacher cosine distillation |

### scripts/

| 파일 | 설명 |
|------|------|
| generate_bbox_phase2.py | Phase 2 구현. boundary start/end score로 proposal ranking 재조정. mAP 출력 |
| eval_perframe_mAP.py | OAD 표준 per-frame mAP 계산 |

### action_detection.py

원본 대비 추가: CausalDSTE / CausalDSTEAux / CausalDSTEAFCM 브랜치,
--lam-distill / --teacher-ckpt / --lam-future argument,
_build_boundary_gt() 공통 함수, teacher feature distillation 로직

---

## Phase 2 설명

**파일:** scripts/generate_bbox_phase2.py

Phase 2는 재학습 없이 기존 모델의 출력(softmax + start/end score)으로 proposal ranking을 개선하는 post-processing이다.
proposal confidence를 아래 식으로 재계산한다:

  score = cls_score^(1-alpha) x (start_score[t_s] x end_score[t_e])^alpha

alpha=0이면 기존 방식과 동일, alpha=0.1이 대부분 실험에서 best.
파라미터 추가 없이 inference 로직만 변경하므로 재학습 불필요.

---

## 실험 이력

### 1단계: Offline - Backbone 탐색

| 방법 | mAP@0.5 | 결과 |
|------|---------|------|
| DSTE + GATv2 | 0.706 | baseline(0.727) 미달, 제외 |
| DSTE + TSM | 0.728 | baseline과 동일, 제외 |

### 2단계: Offline - Head 설계

| 방법 | mAP@0.1 | mAP@0.3 | mAP@0.5 | mAP@0.7 | 결과 |
|------|---------|---------|---------|---------|------|
| DSTE baseline (Phase 0) | 0.856 | 0.825 | 0.727 | 0.469 | 기준점 |
| + aux loss (Phase 1) | 0.843 | 0.812 | 0.723 | 0.472 | 미미한 향상 |
| + Phase 2 | 0.884 | 0.861 | 0.785 | 0.557 | offline best |
| + decouple (Phase 3) | 0.833 | 0.811 | 0.737 | 0.505 | Phase 2 미달, 제외 |
| + STFM | 0.826 | 0.799 | 0.694 | 0.454 | 제외 |
| + CDED | 0.708 | 0.685 | 0.620 | 0.445 | 제외 |

mAPa / mAPv (theta=0.5):
- DSTE offline: 73.7% / 73.4%
- DSTE + Phase 2: 77.7% / 76.7%

### 3단계: Online (Causal) Detection

| 방법 | mAP@0.1 | mAP@0.3 | mAP@0.5 | mAP@0.7 | 비고 |
|------|---------|---------|---------|---------|------|
| CausalDSTE v1 (leaky) | 0.791 | 0.751 | 0.640 | 0.423 | 정보 leaky |
| CausalDSTE v2 (strict) | 0.821 | 0.780 | 0.661 | 0.415 | 완전 causal |
| CausalDSTEAux v1 (leaky) + Phase2 | 0.872 | 0.842 | 0.765 | 0.554 |  |
| CausalDSTEAux v3 (strict) | 0.818 | 0.774 | 0.664 | 0.433 | online baseline |
| CausalDSTEAux v3 + Phase2 | 0.844 | 0.807 | 0.712 | 0.500 | online best |

mAPa / mAPv (theta=0.5):
- CausalDSTEAux v3: 65.1% / 64.1%
- CausalDSTEAux v3 + Phase 2: 71.9% / 71.7%

### 4단계: Online 성능 개선 - AFCM

| 실험 | 설정 | mAP@0.5 (+Phase2) | 결과 |
|------|------|-------------------|------|
| AFCM v1 | k=[2,4,6], lam=0.1 | 70.2% | 미달 |
| AFCM v2 | k=[5,10,15], lam=1.0 | - | loss 폭발 |
| AFCM + offline distill | cosine, lam=0.5 | 71.3% | 동일 수준 |

결론: AFCM 유의미한 개선 없음 

---

## 최종 결과 요약

| 방법 | mAPa | mAPv |
|------|------|------|
| DSTE offline | 73.7% | 73.4% |
| DSTE + Phase 2 (offline best) | 77.7% | 76.7% |
| CausalDSTEAux (online) | 65.1% | 64.1% |
| CausalDSTEAux + Phase 2 (online best) | 71.9% | 71.7% |

---

## 학습 명령어

```bash
# Offline DSTEAux
python action_detection.py --backbone DSTEAux --moda joint \
  --pretrained ./checkpoint/ntu60_xs_j_dste/ntu60_xs_joint_dste.pth.tar \
  --finetune-dataset pku_v1 --protocol cross_subject \
  --lr 0.005 --batch-size 256 --lam 1.0 --sigma 2.0 --tag dste_aux

# Online CausalDSTEAux
python action_detection.py --backbone CausalDSTEAux --moda joint \
  --pretrained ./checkpoint/ntu60_xs_j_dste/ntu60_xs_joint_dste.pth.tar \
  --finetune-dataset pku_v1 --protocol cross_subject \
  --lr 0.005 --batch-size 256 --lam 1.0 --sigma 2.0 --tag causal_dste_aux

# Phase 2 mAP
python scripts/generate_bbox_phase2.py \
  --src results/<exp>/detect_each_frame \
  --gt  scripts/V1_Label \
  --alphas 0.0 0.1 0.2 0.3 \
  --out results/<exp>

# Per-frame mAP
python scripts/eval_perframe_mAP.py \
  --src results/<exp>/detect_each_frame \
  --gt  scripts/V1_Label
```

---

## 환경
Python 3.8 | PyTorch 2.0.0+cu118 | CUDA 11.8 | GPU 4x



