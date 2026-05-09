import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DSTE import DSTE


class DownstreamCDED(nn.Module):
    """
    CDED: Classification + Distance rEgression Detection

    Phase 1 (DSTEAux) 위에 loc_head 추가:
      cls_head:   [B, T, num_class]   # 기존과 동일
      start_head: [B, T, 1]           # boundary score (Phase 1과 동일)
      end_head:   [B, T, 1]           # boundary score (Phase 1과 동일)
      loc_head:   [B, T, 2]           # NEW: [d_start, d_end] regression

    GT:
      foreground frame t in segment [s, e]:
        d_start = t - s  (start까지 거리)
        d_end   = e - t  (end까지 거리)
      background: loc loss 미적용

    Loss:
      L = L_cls + lam*(L_start + L_end) + lam*L_loc
      L_loc = SmoothL1(pred_loc[fg], gt_loc[fg])

    추가 파라미터: Linear(hidden_size, 2) ≈ 2K (전체의 0.004%)
    공정성: backbone/pretrained/dataset 변경 없음
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

        # Phase 1과 동일한 heads
        self.fc         = nn.Linear(self.d_model, num_class)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head   = nn.Linear(hidden_size, 1)

        # NEW: boundary distance regression head
        # 출력: [d_start, d_end] (양수, foreground에서만 의미)
        self.loc_head   = nn.Linear(hidden_size, 2)

    def forward(self, jt, js, bt, bs, mt, ms, knn_eval=False, detect=False):
        if self.modality == 'joint':
            y_t, y_s = self.backbone(jt, js)
        elif self.modality == 'motion':
            y_t, y_s = self.backbone(mt, ms)
        elif self.modality == 'bone':
            y_t, y_s = self.backbone(bt, bs)

        if detect:
            T = y_t.size(1)
            y_s_pool = F.adaptive_avg_pool1d(
                y_s.permute(0, 2, 1), T).permute(0, 2, 1)    # [B, T, C]
            y_i = torch.cat([y_t, y_s_pool], dim=-1)           # [B, T, 2C]

            cls_logits   = self.fc(y_i)                         # [B, T, num_class]
            start_logits = self.start_head(y_t)                 # [B, T, 1]
            end_logits   = self.end_head(y_t)                   # [B, T, 1]
            loc_pred     = self.loc_head(y_t)                   # [B, T, 2]
            # loc_pred[:,:,0] = d_start pred
            # loc_pred[:,:,1] = d_end pred

            return cls_logits, start_logits, end_logits, loc_pred

        # non-detect (recognition)
        y_t_pool = y_t.amax(dim=1)
        y_s_pool = y_s.amax(dim=1)
        y_i = torch.cat([y_t_pool, y_s_pool], dim=-1)
        if knn_eval:
            return y_i
        return self.fc(y_i)
