"""
DSTE_causal_error.py  —  Step 1 (NaN 수정판)
=============================================

수정 핵심:
  - fc는 기존 d_model → num_class 그대로 유지 (pretrained 로드됨)
  - fc_proj를 E_t 전용 작은 gate로 분리
  - E_t는 start_head / end_head 에만 concat (cls는 건드리지 않음)
  - start_head / end_head shape mismatch 해결:
      기존 Linear(hidden, 1) pretrained 로드 후
      E_t는 별도 scalar gate weight로 더함 (shape 호환)

결과: backbone + fc + start/end head 전부 pretrained에서 로드됨
      새 파라미터: e_gate_start (scalar), e_gate_end (scalar) 2개뿐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DSTE_causal import DSTECausal


class FrameDiffErrorModule(nn.Module):
    """E_t = ||z_t - z_{t-1}||² normalized. 파라미터 없음."""
    def forward(self, y_t: torch.Tensor) -> torch.Tensor:
        diff = y_t[:, 1:, :] - y_t[:, :-1, :]         # [B, T-1, H]
        e    = (diff ** 2).sum(dim=-1, keepdim=True)    # [B, T-1, 1]
        pad  = torch.zeros(y_t.size(0), 1, 1, device=y_t.device)
        e    = torch.cat([pad, e], dim=1)               # [B, T, 1]
        emax = e.amax(dim=1, keepdim=True).clamp(min=1e-6)
        return e / emax                                 # [B, T, 1] in [0,1]


class DownstreamCausalError(nn.Module):
    """
    CausalDSTEAux + Latent Frame-Difference Error Signal

    핵심 설계:
      - fc, start_head, end_head: 기존과 완전히 동일한 shape → pretrained 100% 로드
      - E_t는 start/end logit에 learned scalar gate로 더함
        → e_gate_start, e_gate_end: 새 파라미터 각 1개 (0으로 초기화)
        → 학습 초반: E_t 기여 = 0 (기존과 동일), 점점 학습되며 기여
      - NaN 불가능: 새 파라미터가 0 초기화라 초반 출력이 기존 모델과 동일

    detect=True → (cls_logits, start_logits, end_logits)  기존 tuple 그대로
    """
    def __init__(
        self,
        t_input_size: int,
        s_input_size: int,
        hidden_size: int,
        num_head: int,
        num_layer: int,
        num_class: int = 60,
        modality: str = 'joint',
        alpha: float = 0.5,
        gap: int = 4,
        kernel_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.modality = modality
        self.d_model  = 2 * hidden_size

        # ── Backbone: 기존과 완전히 동일 ──────────────────────────
        self.backbone = DSTECausal(
            t_input_size, s_input_size, hidden_size,
            num_head, num_layer,
            alpha=alpha, gap=gap, kernel_size=kernel_size,
        )

        # ── Error module (파라미터 없음) ──────────────────────────
        self.error_module = FrameDiffErrorModule()

        # ── 기존과 동일한 shape → pretrained 완전 로드 ────────────
        self.fc         = nn.Linear(self.d_model, num_class)   # [52, 2048]
        self.start_head = nn.Linear(hidden_size, 1)            # [1, 1024]
        self.end_head   = nn.Linear(hidden_size, 1)            # [1, 1024]

        # ── 새 파라미터: E_t scalar gate (0 초기화 → 안전) ────────
        # start_logit += e_gate_start * E_t
        # 초기값 0 → 학습 초반엔 기존 모델과 완전히 동일
        self.e_gate_start = nn.Parameter(torch.zeros(1))
        self.e_gate_end   = nn.Parameter(torch.zeros(1))

    def forward(self, jt, js, bt, bs, mt, ms,
                knn_eval=False, detect=False):

        # backbone
        if self.modality == 'joint':
            y_t, y_s = self.backbone(jt, js)
        elif self.modality == 'motion':
            y_t, y_s = self.backbone(mt, ms)
        else:
            y_t, y_s = self.backbone(bt, bs)

        # error signal [B, T, 1]
        E_t = self.error_module(y_t)

        if detect:
            T     = y_t.size(1)
            y_s_p = F.adaptive_avg_pool1d(y_s.permute(0,2,1), T).permute(0,2,1)
            y_i   = torch.cat([y_t, y_s_p], dim=-1)   # [B, T, d_model]

            # cls: 기존과 완전히 동일 (E_t 미사용 → pretrained fc 그대로)
            cls_logits = self.fc(y_i)                  # [B, T, num_class]

            # start/end: 기존 logit + learned gate * E_t
            # 초기엔 gate=0이므로 기존과 동일, 점점 E_t 반영
            start_logits = self.start_head(y_t) + self.e_gate_start * E_t
            end_logits   = self.end_head(y_t)   + self.e_gate_end   * E_t

            return cls_logits, start_logits, end_logits

        # recognition
        y_t_p = y_t.amax(dim=1)
        y_s_p = y_s.amax(dim=1)
        y_i   = torch.cat([y_t_p, y_s_p], dim=-1)
        if knn_eval:
            return y_i
        return self.fc(y_i)