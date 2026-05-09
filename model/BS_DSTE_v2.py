import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DSTE import DSTE, DST_Layer


class VGatingDSTLayer(nn.Module):
    """
    t_tr1 대체 레이어.
    V-gating 방식:
        Out = Attn(Q, K, V) * (1 + beta * A[t])
        beta = softplus(alpha) > 0 항상
        alpha: zero-init learnable scalar

    - attn_mask 완전 제거 (shape 문제 없음)
    - A는 gating에 들어갈 때만 detach
    - backbone → mid head auxiliary supervision은 그대로 흐름
    """

    def __init__(self, dst_layer: DST_Layer):
        super().__init__()
        self.dst_layer = dst_layer
        self.alpha = nn.Parameter(torch.zeros(1))  # zero-init

    def forward(self, x, A=None):
        """
        Args:
            x : [B, T, C]
            A : [B, T]  actionness (maximum(start, end)), NOT detached yet
                None이면 vanilla DST_Layer와 동일
        Returns:
            [B, T, C]
        """
        if A is None:
            return self.dst_layer(x)

        dst = self.dst_layer
        alpha_dst = dst.alpha
        beta_dst  = dst.beta

        # ── CA path (V-gating 적용) ───────────────────────────────────────
        ca = dst.CA
        Fseq2    = x.permute(0, 2, 1)
        Ftc      = (Fseq2 + ca.act(ca.conv1(Fseq2))).permute(0, 2, 1)
        Ftc_norm = ca.norm1(Ftc)

        # vanilla attention (mask 없음)
        attn_out, _ = ca.attn(Ftc_norm, Ftc_norm, Ftc_norm)
        Ftc = ca.drop(attn_out) + x

        # V-gating: Out * (1 + beta * A_det[t])
        beta    = F.softplus(self.alpha)              # 항상 양수
        A_det   = A.detach()                          # gating에만 stop-gradient
        gate    = 1.0 + beta * A_det.unsqueeze(-1)    # [B, T, 1]
        Ftc     = Ftc * gate

        ca_out = ca.drop(ca.mlp(ca.norm2(Ftc)))

        # ── DSA path (vanilla) ───────────────────────────────────────────
        dsa_out = dst.DSA(x)

        return alpha_dst * ca_out + beta_dst * dsa_out


class DownstreamBSv2(nn.Module):
    """
    BS-DSTE v2: Boundary Refinement Loop

    구조:
        DSTE backbone
            ↓
        t_tr (layer 1, generic)
            ↓
        [Mid] coarse start_head_mid, end_head_mid
              A[t] = maximum(start_mid[t], end_mid[t])
            ↓
        t_tr1 → VGatingDSTLayer (V-gating with A)
        s_tr1 (건드리지 않음)
            ↓
        [Late] refined start_head2, end_head2
        final cls_head, start_head, end_head

    Refinement loop:
        coarse boundary (mid) → gating → refined boundary (late)
        Phase 2 proposal에서 start2/end2 사용 → high-IoU gain 기대

    추가 파라미터:
        start_head_mid, end_head_mid : Linear(C, 1) × 2
        start_head2, end_head2       : Linear(C, 1) × 2
        alpha                        : scalar (zero-init)
        합계 ~4K params (전체 57M의 0.007%)
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

        self.backbone = DSTE(
            t_input_size, s_input_size, hidden_size,
            num_head, num_layer,
            alpha=alpha, gap=gap, kernel_size=kernel_size,
        )

        # backbone.t_tr1 참조 (파라미터 공유, 중복 카운팅 없음)
        self.guided_t_tr1 = VGatingDSTLayer(self.backbone.t_tr1)

        # Mid coarse boundary heads
        self.start_head_mid = nn.Linear(hidden_size, 1)
        self.end_head_mid   = nn.Linear(hidden_size, 1)

        # Late refined boundary heads (t_tr1 이후 feature 사용)
        self.start_head2 = nn.Linear(hidden_size, 1)
        self.end_head2   = nn.Linear(hidden_size, 1)

        # Final heads
        self.cls_head   = nn.Linear(hidden_size * 2, num_class)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head   = nn.Linear(hidden_size, 1)

    def forward(self, jt, js, bt, bs, mt, ms, knn_eval=False, detect=False):
        if self.modality == 'joint':
            jt_in, js_in = jt, js
        elif self.modality == 'motion':
            jt_in, js_in = mt, ms
        elif self.modality == 'bone':
            jt_in, js_in = bt, bs

        t_src, s_src = self.backbone.ske_emb(jt_in, js_in)
        B = t_src.size(0)

        # ── layer 1 ──────────────────────────────────────────────────────
        y_t1 = self.backbone.t_tr(self.backbone.tpe(t_src))
        y_s1 = self.backbone.s_tr(s_src + self.backbone.spe.expand(B, -1, -1))

        # ── Mid: coarse boundary → Actionness ────────────────────────────
        A_start_logit = self.start_head_mid(y_t1).squeeze(-1)  # [B, T] logit
        A_end_logit   = self.end_head_mid(y_t1).squeeze(-1)    # [B, T] logit
        A_start_mid   = torch.sigmoid(A_start_logit)
        A_end_mid     = torch.sigmoid(A_end_logit)
        A = ((A_start_mid + A_end_mid) / 2) ** 2

        # ── layer 2 (V-gating temporal, vanilla spatial) ─────────────────
        y_t2 = self.guided_t_tr1(self.backbone.tpe(y_t1), A=A)
        y_s2 = self.backbone.s_tr1(y_s1 + self.backbone.spe.expand(B, -1, -1))

        if detect:
            T = y_t2.size(1)
            y_s_pool = y_s2.mean(dim=1, keepdim=True).repeat(1, T, 1)
            y_i = torch.cat([y_t2, y_s_pool], dim=-1)

            cls_logits    = self.cls_head(y_i)         # [B, T, num_class]
            start_logits  = self.start_head(y_t2)      # [B, T, 1]
            end_logits    = self.end_head(y_t2)        # [B, T, 1]
            start_logits2 = self.start_head2(y_t2)    # [B, T, 1]
            end_logits2   = self.end_head2(y_t2)      # [B, T, 1]

            return (cls_logits, start_logits, end_logits,
                    A_start_logit, A_end_logit,
                    A_start_mid, A_end_mid,
                    start_logits2, end_logits2)

        # ── non-detect ───────────────────────────────────────────────────
        y_t_pool = y_t2.amax(dim=1)
        y_s_pool = y_s2.amax(dim=1)
        y_i = torch.cat([y_t_pool, y_s_pool], dim=-1)
        if knn_eval:
            return y_i
        return self.cls_head(y_i)
