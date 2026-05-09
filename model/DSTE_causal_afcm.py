"""
DSTE_causal_afcm.py — CausalDSTEAux + AFCM + Feature Distillation
──────────────────────────────────────────────────────────────────
학습 시:
  - Frozen offline teacher (DownstreamAux)의 y_t를 추출
  - AFCM output과 cosine similarity loss로 distillation
  - L_total = L_cls + L_bnd + λ_distill * L_distill

추론 시:
  - teacher 없음, student만 동작 → 완전 causal

Collapse 방지:
  - teacher.detach() → stop-gradient
  - cosine similarity loss (방향만, magnitude 무관)
  - teacher 완전 frozen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DSTE_causal import DSTECausal
from model.AFCM import AFCM


class DownstreamCausalAFCM(nn.Module):
    def __init__(
        self,
        t_input_size,
        s_input_size,
        hidden_size,
        num_head,
        num_layer,
        num_class   = 52,
        modality    = 'joint',
        alpha       = 0.5,
        gap         = 4,
        kernel_size = 1,
        k_steps     = [2, 4, 6],
        lam_future  = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.modality   = modality
        self.d_model    = 2 * hidden_size
        self.lam_future = lam_future

        self.backbone = DSTECausal(
            t_input_size, s_input_size, hidden_size,
            num_head, num_layer,
            alpha=alpha, gap=gap, kernel_size=kernel_size,
        )
        self.afcm = AFCM(
            hidden_size=hidden_size,
            num_head=num_head,
            k_steps=k_steps,
        )
        self.fc         = nn.Linear(self.d_model, num_class)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head   = nn.Linear(hidden_size, 1)

    def forward(self, jt, js, bt, bs, mt, ms,
                knn_eval=False, detect=False,
                compute_future_loss=False,
                teacher_y_t=None):
        """
        Args:
            compute_future_loss: True → (cls, start, end, L_distill) 반환
            teacher_y_t: [B, T, H] offline teacher의 y_t (학습 시만)
        """
        if self.modality == 'joint':
            y_t, y_s = self.backbone(jt, js)
        elif self.modality == 'motion':
            y_t, y_s = self.backbone(mt, ms)
        elif self.modality == 'bone':
            y_t, y_s = self.backbone(bt, bs)

        # ── AFCM ──────────────────────────────────────────────────────
        # future prediction loss는 더 이상 사용 안 함 (distillation으로 대체)
        y_t_fused = self.afcm(y_t, compute_loss=False)

        # ── Distillation loss (학습 시만) ──────────────────────────────
        if compute_future_loss and teacher_y_t is not None:
            # cosine similarity loss: collapse 방지
            # teacher_y_t: [B, T, H] (detach는 호출부에서)
            cos_sim = F.cosine_similarity(
                y_t_fused.reshape(-1, y_t_fused.size(-1)),
                teacher_y_t.reshape(-1, teacher_y_t.size(-1)),
                dim=-1
            )
            L_distill = (1.0 - cos_sim).mean()
        else:
            L_distill = None

        # ── detect mode ───────────────────────────────────────────────
        if detect:
            T = y_t_fused.size(1)
            y_s_pool = F.adaptive_avg_pool1d(
                y_s.permute(0, 2, 1), T).permute(0, 2, 1)
            y_i = torch.cat([y_t_fused, y_s_pool], dim=-1)
            cls_logits   = self.fc(y_i)
            start_logits = self.start_head(y_t_fused)
            end_logits   = self.end_head(y_t_fused)

            if compute_future_loss and L_distill is not None:
                return cls_logits, start_logits, end_logits, L_distill
            return cls_logits, start_logits, end_logits

        # ── recognition mode ──────────────────────────────────────────
        y_t_pool = y_t_fused.amax(dim=1)
        y_s_pool = y_s.amax(dim=1)
        y_i = torch.cat([y_t_pool, y_s_pool], dim=-1)
        if knn_eval:
            return y_i
        return self.fc(y_i)


class TeacherWrapper(nn.Module):
    """
    Offline DSTEAux를 frozen teacher로 래핑.
    forward 시 y_t만 반환.
    추론 시엔 사용 안 함.
    """
    def __init__(self, teacher_model):
        super().__init__()
        self.teacher = teacher_model
        # 완전 frozen
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_feature(self, jt, js, bt, bs, mt, ms, modality='joint'):
        """y_t [B, T, H] 반환 (stop-gradient 보장)"""
        if modality == 'joint':
            y_t, y_s = self.teacher.backbone(jt, js)
        elif modality == 'motion':
            y_t, y_s = self.teacher.backbone(mt, ms)
        elif modality == 'bone':
            y_t, y_s = self.teacher.backbone(bt, bs)
        return y_t.detach()