import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DSTE import DSTE


class DownstreamAux(nn.Module):
    """Phase 1: DSTE + cls_head (identical to baseline) + auxiliary start/end heads.

    detect=True path:
        DSTE → y_t [B, T, C], y_s [B, V, C]
        y_s adaptive_avg_pool → [B, T, C]
        y_i = concat(y_t, y_s) → [B, T, 2C]
        cls_head(y_i)   → [B, T, num_class]   (identical to Phase 0)
        start_head(y_t) → [B, T, 1]            (auxiliary, training only)
        end_head(y_t)   → [B, T, 1]            (auxiliary, training only)

    detect=False path: identical to Phase 0 Downstream.

    Proposal generation at inference: identical to Phase 0 (cls softmax threshold).
    Boundary heads are NOT used at inference — pure auxiliary supervision.
    """

    def __init__(
        self,
        t_input_size,
        s_input_size,
        hidden_size,
        num_head,
        num_layer,
        num_class=60,
        modality='joint',
        alpha=0.5,
        gap=4,
        kernel_size=1,
        **kwargs,
    ):
        super().__init__()
        self.modality = modality
        self.d_model  = 2 * hidden_size

        self.backbone = DSTE(
            t_input_size, s_input_size, hidden_size,
            num_head, num_layer,
            alpha=alpha, gap=gap, kernel_size=kernel_size,
        )

        self.fc          = nn.Linear(self.d_model, num_class)   # identical name to baseline
        self.start_head  = nn.Linear(hidden_size, 1)
        self.end_head    = nn.Linear(hidden_size, 1)

    def forward(self, jt, js, bt, bs, mt, ms, knn_eval=False, detect=False):
        if self.modality == 'joint':
            y_t, y_s = self.backbone(jt, js)
        elif self.modality == 'motion':
            y_t, y_s = self.backbone(mt, ms)
        elif self.modality == 'bone':
            y_t, y_s = self.backbone(bt, bs)

        if detect:
            T = y_t.size(1)
            y_s_pool = F.adaptive_avg_pool1d(y_s.permute(0, 2, 1), T).permute(0, 2, 1)
            y_i = torch.cat([y_t, y_s_pool], dim=-1)          # [B, T, 2C]
            cls_logits   = self.fc(y_i)                        # [B, T, num_class]
            start_logits = self.start_head(y_t)                # [B, T, 1]
            end_logits   = self.end_head(y_t)                  # [B, T, 1]
            return cls_logits, start_logits, end_logits

        y_t_pool = y_t.amax(dim=1)
        y_s_pool = y_s.amax(dim=1)
        y_i = torch.cat([y_t_pool, y_s_pool], dim=-1)
        if knn_eval:
            return y_i
        return self.fc(y_i)
