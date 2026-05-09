import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DSTE import DSTE


def temporal_shift(x, fold_div=8):
    """In-place-safe TSM shift on temporal stream.

    Args:
        x        : [B, T, C]
        fold_div : split denominator; fold = C // fold_div channels shifted

    Returns:
        out : [B, T, C]
            [:, 1:,    :fold]   ← past   (shift forward)
            [:, :-1, fold:2f]   ← future (shift backward)
            [:, :,   2f:   ]   ← unchanged
    """
    B, T, C = x.shape
    fold = C // fold_div

    out = torch.zeros_like(x)
    # past channel — shift forward by 1 frame
    out[:, 1:,    :fold]       = x[:, :-1, :fold]
    # future channel — shift backward by 1 frame
    out[:, :-1,   fold:2*fold] = x[:, 1:,  fold:2*fold]
    # residual channels — unchanged
    out[:, :,     2*fold:]     = x[:, :,   2*fold:]
    return out


class TSMBlock(nn.Module):
    """Pure Temporal Shift Module (TSM paper, Lin et al. 2019).

    No norm, no MLP — pure channel shift along the temporal axis + residual.
        x = x + shift(x)

    Shape:
        Input  : [B, T, C]
        Output : [B, T, C]
    """

    def __init__(self, shift_div=8):
        super().__init__()
        self.shift_div = shift_div

    def forward(self, x):
        # x : [B, T, C]
        return x + temporal_shift(x, fold_div=self.shift_div)


class DownstreamTSM(nn.Module):
    """Downstream detection model with TSM temporal re-encoding after DSTE.

    Architecture
    ------------
    DSTE (pretrained backbone)
        → y_t [B, 64, C]          temporal stream
        → y_s [B, 50, C]          spatial stream (unchanged)
              ↓
          TSMBlock                 temporal boundary refinement on y_t
              ↓
         y_t_refined [B, 64, C]
              ↓
    adaptive_avg_pool1d(y_s) → [B, 64, C]
    concat(y_t_refined, y_s) → [B, 64, 2C]
    fc → [B, 64, num_class]       frame-wise prediction (same as baseline)

    TSMBlock has no learnable parameters (pure shift).
    DSTE backbone is loaded from pretrained checkpoint.
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
        shift_div=8,
        **kwargs,
    ):
        super().__init__()
        self.modality = modality
        self.d_model  = 2 * hidden_size

        # ── DSTE backbone (never modified) ────────────────────────────────
        self.backbone = DSTE(
            t_input_size, s_input_size, hidden_size,
            num_head, num_layer,
            alpha=alpha, gap=gap, kernel_size=kernel_size,
        )

        # ── TSM temporal re-encoding (no learnable params) ───────────────
        self.tsm = TSMBlock(shift_div=shift_div)

        # ── Detection head (identical to Downstream baseline) ─────────────
        self.fc = nn.Linear(self.d_model, num_class)

    def forward(self, jt, js, bt, bs, mt, ms, knn_eval=False, detect=False):
        """
        Inputs (PKU/NTU detection):
            jt : [B, T=64, M*V*C=150]
            js : [B, M*V=50, T*C=192]
            bt, bs, mt, ms : same shapes (bone / motion modalities)

        detect=True  → [B, T=64, num_class]
        detect=False → [B, num_class]
        """
        # ── 1. DSTE backbone ──────────────────────────────────────────────
        if self.modality == 'joint':
            y_t, y_s = self.backbone(jt, js)    # [B,64,C], [B,50,C]
        elif self.modality == 'motion':
            y_t, y_s = self.backbone(mt, ms)
        elif self.modality == 'bone':
            y_t, y_s = self.backbone(bt, bs)

        # ── 2. TSM temporal re-encoding ───────────────────────────────────
        # y_t : [B, 64, C]
        y_t = self.tsm(y_t)                                   # [B, 64, C]

        # ── 3. Head ───────────────────────────────────────────────────────
        if detect:
            y_s_t = F.adaptive_avg_pool1d(
                y_s.permute(0, 2, 1), y_t.size(1)   # [B, C, 50] → [B, C, 64]
            ).permute(0, 2, 1)                        # [B, 64, C]
            y_i = torch.cat([y_t, y_s_t], dim=-1)    # [B, 64, 2C]
            return self.fc(y_i)                       # [B, 64, num_class]

        y_t_pool = y_t.amax(dim=1)                   # [B, C]
        y_s_pool = y_s.amax(dim=1)                   # [B, C]
        y_i = torch.cat([y_t_pool, y_s_pool], dim=-1)
        if knn_eval:
            return y_i
        return self.fc(y_i)
