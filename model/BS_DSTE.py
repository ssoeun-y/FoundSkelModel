import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DSTE import DSTE, DST_Layer


class BoundaryGuidedDSTLayer(nn.Module):
    """
    t_tr1을 대체하는 레이어.
    vanilla DST_Layer와 구조 동일 + boundary-conditioned attention bias 주입.

    attn_mask = alpha * M_bias
        alpha  : zero-init learnable scalar
        M_bias : (B*H, T, T),  M_bias[b,i,j] = A[b,i] * A[b,j],  detach()

    alpha만 추가 파라미터 (scalar 1개).
    Q/K/V 구조, MLP, DSA, CA 전혀 건드리지 않음.
    """

    def __init__(self, dst_layer: DST_Layer):
        super().__init__()
        self.dst_layer = dst_layer
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, A=None):
        """
        Args:
            x : [B, T, C]
            A : [B, T]  actionness score (sigmoid(start) * sigmoid(end)), detached
                None이면 vanilla DST_Layer와 동일하게 동작
        Returns:
            [B, T, C]
        """
        if A is None:
            return self.dst_layer(x)

        B, T, C = x.shape

        M_bias = (A.unsqueeze(2) * A.unsqueeze(1)).detach()  # (B, T, T)
        bias = self.alpha * M_bias                            # (B, T, T)

        # ── DST_Layer 내부 재현: CA(bias 주입) + DSA(vanilla) ─────────────────
        dst       = self.dst_layer
        alpha_dst = dst.alpha
        beta_dst  = dst.beta

        # CA path — F.multi_head_attention_forward 직접 호출 (B, T, T) per-sample bias
        ca = dst.CA
        Fseq2    = x.permute(0, 2, 1)
        Ftc      = (Fseq2 + ca.act(ca.conv1(Fseq2))).permute(0, 2, 1)
        Ftc_norm = ca.norm1(Ftc)
        q = k = v = Ftc_norm
        attn_out, _ = F.multi_head_attention_forward(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            embed_dim_to_check=ca.attn.embed_dim,
            num_heads=ca.attn.num_heads,
            in_proj_weight=ca.attn.in_proj_weight,
            in_proj_bias=ca.attn.in_proj_bias,
            bias_k=None, bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=ca.attn.out_proj.weight,
            out_proj_bias=ca.attn.out_proj.bias,
            training=self.training,
            attn_mask=bias,
        )
        attn_out = attn_out.transpose(0, 1)  # (B, T, C)
        Ftc    = ca.drop(attn_out) + x
        ca_out = ca.drop(ca.mlp(ca.norm2(Ftc)))

        # DSA path — vanilla (bias 없음)
        dsa_out = dst.DSA(x)

        return alpha_dst * ca_out + beta_dst * dsa_out


class DownstreamBS(nn.Module):
    """
    BS-DSTE: Boundary-Specialized DSTE

    구조:
        DSTE backbone (pretrained weight 그대로 로드)
            ↓
        t_tr  (layer 1, generic)
            ↓
        [Mid] coarse start_head_mid, end_head_mid  → A[t]
            ↓
        t_tr1 → BoundaryGuidedDSTLayer (attn_mask = alpha * M_bias)
        s_tr1 (건드리지 않음)
            ↓
        final cls_head, start_head, end_head

    추가 파라미터:
        - start_head_mid: Linear(C, 1)
        - end_head_mid:   Linear(C, 1)
        - alpha:          scalar (zero-init)
        합계 ~2K params (전체 57M의 0.004%)
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

        # backbone.t_tr1의 deepcopy → pretrained weight 로드 후 별도 객체로 분리
        self.guided_t_tr1 = BoundaryGuidedDSTLayer(self.backbone.t_tr1)

        # Mid coarse boundary heads (t_tr 이후, hidden_size=C)
        self.start_head_mid = nn.Linear(hidden_size, 1)
        self.end_head_mid   = nn.Linear(hidden_size, 1)

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

        # ── layer 1 ──────────────────────────────────────────────────────────
        y_t1 = self.backbone.t_tr(self.backbone.tpe(t_src))
        y_s1 = self.backbone.s_tr(s_src + self.backbone.spe.expand(B, -1, -1))

        # ── Mid: coarse boundary → Actionness ────────────────────────────────
        A_start = torch.sigmoid(self.start_head_mid(y_t1).squeeze(-1))  # [B, T]
        A_end   = torch.sigmoid(self.end_head_mid(y_t1).squeeze(-1))    # [B, T]
        temp = 5.0
        tau  = 0.5
        A = torch.sigmoid(temp * (A_start + A_end - tau))

        # ── layer 2 (boundary-guided temporal, vanilla spatial) ───────────────
        y_t2 = self.guided_t_tr1(self.backbone.tpe(y_t1), A=A)
        y_s2 = self.backbone.s_tr1(y_s1 + self.backbone.spe.expand(B, -1, -1))

        if detect:
            T = y_t2.size(1)
            y_s_pool = y_s2.mean(dim=1, keepdim=True).repeat(1, T, 1)
            y_i = torch.cat([y_t2, y_s_pool], dim=-1)

            cls_logits   = self.cls_head(y_i)
            start_logits = self.start_head(y_t2)
            end_logits   = self.end_head(y_t2)

            return cls_logits, start_logits, end_logits, A_start, A_end

        # ── non-detect ────────────────────────────────────────────────────────
        y_t_pool = y_t2.amax(dim=1)
        y_s_pool = y_s2.amax(dim=1)
        y_i = torch.cat([y_t_pool, y_s_pool], dim=-1)
        if knn_eval:
            return y_i
        return self.cls_head(y_i)
