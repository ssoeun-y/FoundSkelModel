import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DSTE import DSTE


class DownstreamSTFM(nn.Module):
    """
    STFM: Skeleton Temporal Field Modeling

    action segment를 continuous transition field [-1, +1]로 모델링.
    boundary = field gradient:
        start_from_field = sigmoid(k * (F[t+1] - F[t]))  ← positive gradient
        end_from_field   = sigmoid(k * (F[t] - F[t+1]))  ← negative gradient

    추가 파라미터: field_head Linear(C, 1) ≈ 1K (전체의 0.002%)
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
        field_k=5.0,
        **kwargs,
    ):
        super().__init__()
        self.modality = modality
        self.field_k  = field_k

        self.backbone = DSTE(
            t_input_size, s_input_size, hidden_size,
            num_head, num_layer, alpha=alpha, gap=gap, kernel_size=kernel_size,
        )

        self.field_head = nn.Linear(hidden_size, 1)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head   = nn.Linear(hidden_size, 1)
        self.cls_head   = nn.Linear(hidden_size * 2, num_class)

    def forward(self, jt, js, bt, bs, mt, ms, knn_eval=False, detect=False):
        if self.modality == 'joint':
            jt_in, js_in = jt, js
        elif self.modality == 'motion':
            jt_in, js_in = mt, ms
        elif self.modality == 'bone':
            jt_in, js_in = bt, bs

        t_src, s_src = self.backbone.ske_emb(jt_in, js_in)
        B = t_src.size(0)

        y_t1 = self.backbone.t_tr(self.backbone.tpe(t_src))
        y_s1 = self.backbone.s_tr(s_src + self.backbone.spe.expand(B, -1, -1))
        y_t2 = self.backbone.t_tr1(self.backbone.tpe(y_t1))
        y_s2 = self.backbone.s_tr1(y_s1 + self.backbone.spe.expand(B, -1, -1))

        if detect:
            T = y_t2.size(1)
            y_s_pool = y_s2.mean(dim=1, keepdim=True).repeat(1, T, 1)
            y_i = torch.cat([y_t2, y_s_pool], dim=-1)

            cls_logits   = self.cls_head(y_i)         # [B, T, num_class]
            start_logits = self.start_head(y_t2)      # [B, T, 1]
            end_logits   = self.end_head(y_t2)        # [B, T, 1]

            F_pred = torch.tanh(self.field_head(y_t2))          # [B, T, 1]
            F_sq   = F_pred.squeeze(-1)                          # [B, T]
            k = self.field_k
            start_from_field = torch.sigmoid(k * (F_sq[:, 1:] - F_sq[:, :-1]))  # [B, T-1]
            end_from_field   = torch.sigmoid(k * (F_sq[:, :-1] - F_sq[:, 1:]))  # [B, T-1]

            return (cls_logits, start_logits, end_logits,
                    F_pred, start_from_field, end_from_field)

        y_t_pool = y_t2.amax(dim=1)
        y_s_pool = y_s2.amax(dim=1)
        y_i = torch.cat([y_t_pool, y_s_pool], dim=-1)
        if knn_eval:
            return y_i
        return self.cls_head(y_i)
