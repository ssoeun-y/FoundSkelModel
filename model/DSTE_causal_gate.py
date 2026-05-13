import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DSTE_causal import DSTECausal


class DownstreamCausalGate(nn.Module):
    def __init__(self, t_input_size, s_input_size, hidden_size,
                 num_head, num_layer, num_class=52, modality='joint',
                 alpha=0.5, gap=4, kernel_size=1, **kwargs):
        super().__init__()
        self.modality = modality
        self.d_model  = 2 * hidden_size
        self.backbone = DSTECausal(
            t_input_size, s_input_size, hidden_size,
            num_head, num_layer, alpha=alpha, gap=gap, kernel_size=kernel_size,
        )
        self.gate_head  = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.fc         = nn.Linear(self.d_model, num_class)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head   = nn.Linear(hidden_size, 1)

    def forward(self, jt, js, bt, bs, mt, ms, knn_eval=False, detect=False):
        if self.modality == 'joint':
            y_t, y_s = self.backbone(jt, js)
        elif self.modality == 'motion':
            y_t, y_s = self.backbone(mt, ms)
        elif self.modality == 'bone':
            y_t, y_s = self.backbone(bt, bs)

        gate     = self.gate_head(y_t)   # [B, T, 1]
        y_t_gate = y_t * gate            # background 억제

        if detect:
            T = y_t_gate.size(1)
            y_s_pool = F.adaptive_avg_pool1d(y_s.permute(0,2,1), T).permute(0,2,1)
            y_i = torch.cat([y_t_gate, y_s_pool], dim=-1)
            cls_logits   = self.fc(y_i)
            start_logits = self.start_head(y_t_gate)
            end_logits   = self.end_head(y_t_gate)
            return cls_logits, start_logits, end_logits, gate

        y_t_pool = y_t_gate.amax(dim=1)
        y_s_pool = y_s.amax(dim=1)
        y_i = torch.cat([y_t_pool, y_s_pool], dim=-1)
        if knn_eval:
            return y_i
        return self.fc(y_i)
