import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DSTE import DSTE, trunc_normal_

# NTU/PKU 25-joint skeleton connectivity (1-indexed)
_BONE = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
    (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12),
]


def build_adj(num_joints=25, num_persons=2):
    """Build binary adjacency matrix for the skeleton graph.

    V_total = num_joints * num_persons  (50 for 2-person NTU/PKU)

    Edges:
      - Self-loops for every node
      - Bidirectional skeleton edges per person (no inter-person edges)

    Returns:
        adj : FloatTensor [V_total, V_total]  e.g. [50, 50]
    """
    V = num_joints * num_persons
    adj = torch.zeros(V, V)

    for i in range(V):
        adj[i, i] = 1.0

    for v1, v2 in _BONE:
        for p in range(num_persons):
            i = (v1 - 1) + p * num_joints
            j = (v2 - 1) + p * num_joints
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    return adj


class GATv2Layer(nn.Module):
    """GATv2 — Dynamic Graph Attention Layer.

    Reference: "How Attentive are Graph Attention Networks?" (Brody et al., ICLR 2022)

    GAT  vs GATv2:
        GAT  : e_ij = a^T [ W·h_i  ||  W·h_j ]          (static  — decomposable)
        GATv2: e_ij = a^T   LeakyReLU( W_l·h_i + W_r·h_j ) (dynamic — true interaction)

    Shape flow:
        x          [B, V, C]   input
        W_l(x)     [B, V, D]   source projection   (D = attn_dim)
        W_r(x)     [B, V, D]   target projection
        unsqueeze+broadcast:
          W_l(x).unsqueeze(2)  [B, V, 1, D]
        + W_r(x).unsqueeze(1)  [B, 1, V, D]
        = e_raw               [B, V, V, D]   ← true i,j interaction
        LeakyReLU(e_raw)      [B, V, V, D]
        a(...)                [B, V, V, 1]
        squeeze               [B, V, V]
        mask + softmax → alpha [B, V, V]
        W_out(x) = Wh         [B, V, C]
        bmm(alpha, Wh)        [B, V, C]
        ELU                   [B, V, C]

    Memory (attn_dim=64, V=50, B=256):
        [B, V, V, D] = 256×50×50×64×4 bytes ≈ 164 MB  (acceptable)
    """

    def __init__(self, in_features, out_features, attn_dim=64, dropout=0.1):
        super().__init__()
        self.W_l   = nn.Linear(in_features,  attn_dim,     bias=False)  # source projection
        self.W_r   = nn.Linear(in_features,  attn_dim,     bias=False)  # target projection
        self.a     = nn.Linear(attn_dim,      1,            bias=False)  # attention scalar
        self.W_out = nn.Linear(in_features,  out_features, bias=False)  # value projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x   : [B, V, in_features]
        adj : [V, V]  binary (1=edge, 0=no edge), same device as x
        """
        B, V, _ = x.shape

        # ── value projection (what gets aggregated) ───────────────────────
        Wh = self.W_out(x)                             # [B, V, out_features]

        # ── GATv2 dynamic attention ───────────────────────────────────────
        e_l = self.W_l(x)                              # [B, V, attn_dim]
        e_r = self.W_r(x)                              # [B, V, attn_dim]

        # [B, V, 1, D] + [B, 1, V, D] → [B, V, V, D]
        # e[b,i,j] = W_l·h_i + W_r·h_j  (joint interaction of i and j)
        e = F.leaky_relu(
            e_l.unsqueeze(2) + e_r.unsqueeze(1),       # [B, V, V, attn_dim]
            negative_slope=0.2,
        )
        e = self.a(e).squeeze(-1)                      # [B, V, V]

        # ── mask non-edges → -inf before softmax ──────────────────────────
        mask = (adj == 0).unsqueeze(0)                 # [1, V, V]
        e = e.masked_fill(mask, float('-inf'))

        alpha = F.softmax(e, dim=2)                    # [B, V, V]  (over neighbors dim)
        alpha = self.dropout(alpha)

        # ── aggregate ─────────────────────────────────────────────────────
        out = torch.bmm(alpha, Wh)                     # [B, V, out_features]
        return F.elu(out)


class DownstreamGAT(nn.Module):
    """Downstream detection model: DSTE backbone + temporal-conditioned GATv2.

    Full shape flow (PKU v1, hidden_size=1024):
    ─────────────────────────────────────────────────────────
    Input
        jt  [B, T=64, M*V*C=150]
        js  [B, M*V=50, T*C=192]

    DSTE backbone (frozen)
        y_t [B, 64, 1024]   temporal stream
        y_s [B, 50, 1024]   spatial stream

    Step 1 — Cross-attention: spatial ← temporal
        query = y_s  [B, 50, 1024]
        key/value = y_t  [B, 64, 1024]
        → y_s_cond  [B, 50, 1024]   (each joint attends to relevant frames)
        y_s = y_s + y_s_cond        (residual)

    Step 2 — GATv2 spatial re-encoding
        LayerNorm(y_s)              [B, 50, 1024]
        → y_s_refined               [B, 50, 1024]
        y_s = y_s + y_s_refined     (residual)

    Step 3 — Cross-attention: temporal ← spatial
        query = y_t  [B, 64, 1024]
        key/value = y_s  [B, 50, 1024]
        → y_t_enriched  [B, 64, 1024]   (each frame absorbs spatial context)
        y_t = y_t + y_t_enriched        (residual)

    Detection head
        concat(y_t, y_s.amax(1).expand)  [B, 64, 2048]
        fc                               [B, 64,   52]

    NOTE: TSM will be inserted between Step 3 and Detection head.
    ─────────────────────────────────────────────────────────
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
        num_joints=25,
        num_persons=2,
        attn_dim=64,
        gat_dropout=0.1,
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

        # ── GATv2 spatial re-encoding ─────────────────────────────────────
        self.gat_norm = nn.LayerNorm(hidden_size)
        self.gat      = GATv2Layer(
            in_features=hidden_size,
            out_features=hidden_size,
            attn_dim=attn_dim,
            dropout=gat_dropout,
        )

        # ── Cross-attention 1: spatial ← temporal ─────────────────────────
        # query=y_s [B,V,C], key/value=y_t [B,T,C]
        # each joint selectively attends to relevant frames
        self.cross_attn_s   = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_norm_s     = nn.LayerNorm(hidden_size)  # for y_s (query)
        self.cross_norm_t_key = nn.LayerNorm(hidden_size)  # for y_t (key/value in cross_attn_s)

        # ── Cross-attention 2: temporal ← spatial ─────────────────────────
        # query=y_t [B,T,C], key/value=y_s [B,V,C]
        # each frame absorbs spatially-refined joint context
        self.cross_attn_t   = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_norm_t   = nn.LayerNorm(hidden_size)  # for y_t (query)
        self.cross_norm_s_key = nn.LayerNorm(hidden_size)  # for y_s (key/value in cross_attn_t)

        # ── Detection head ────────────────────────────────────────────────
        self.fc = nn.Linear(self.d_model, num_class)

        # ── Skeleton adjacency — registered buffer (auto to GPU) ──────────
        adj = build_adj(num_joints=num_joints, num_persons=num_persons)
        self.register_buffer('adj', adj)

        self._init_new_weights()

    def _init_new_weights(self):
        """Init only the new weights (GAT + cross-attention norms).
        fc is initialised externally in action_detection.py (normal_ / zero_).
        """
        for w in [self.gat.W_l.weight, self.gat.W_r.weight,
                  self.gat.a.weight,   self.gat.W_out.weight]:
            trunc_normal_(w, std=0.02)
        nn.init.ones_(self.gat_norm.weight)
        nn.init.zeros_(self.gat_norm.bias)
        nn.init.ones_(self.cross_norm_s.weight)
        nn.init.zeros_(self.cross_norm_s.bias)
        nn.init.ones_(self.cross_norm_t_key.weight)
        nn.init.zeros_(self.cross_norm_t_key.bias)
        nn.init.ones_(self.cross_norm_t.weight)
        nn.init.zeros_(self.cross_norm_t.bias)
        nn.init.ones_(self.cross_norm_s_key.weight)
        nn.init.zeros_(self.cross_norm_s_key.bias)

    def forward(self, jt, js, bt, bs, mt, ms, knn_eval=False, detect=False):
        # ── 1. DSTE backbone ──────────────────────────────────────────────
        if self.modality == 'joint':
            y_t, y_s = self.backbone(jt, js)   # [B,64,1024], [B,50,1024]
        elif self.modality == 'motion':
            y_t, y_s = self.backbone(mt, ms)
        elif self.modality == 'bone':
            y_t, y_s = self.backbone(bt, bs)

        # ── 2. Cross-attention: y_s가 y_t를 참조 ─────────────────────────
        # 각 joint가 관련 있는 시점의 temporal context를 선택적으로 흡수
        # query=y_s [B,V,C], key/value=y_t [B,T,C]
        y_s_cond, _ = self.cross_attn_s(
            query=self.cross_norm_s(y_s),         # [B, 50, 1024]  y_s 전용 norm
            key=self.cross_norm_t_key(y_t),        # [B, 64, 1024]  y_t 전용 norm
            value=self.cross_norm_t_key(y_t),      # [B, 64, 1024]  key와 동일 norm
        )
        y_s = y_s + y_s_cond                # [B, 50, 1024]  residual

        # ── 3. GATv2 spatial re-encoding (pre-norm + residual) ───────────
        # temporally-conditioned y_s 위에서 joint 간 관계 정리
        y_s = y_s + self.gat(self.gat_norm(y_s), self.adj)   # [B, 50, 1024]

        # ── 4. Cross-attention: y_t가 y_s를 참조 ─────────────────────────
        # 각 프레임이 spatially-refined joint context를 흡수
        # query=y_t [B,T,C], key/value=y_s [B,V,C]
        y_t_enriched, _ = self.cross_attn_t(
            query=self.cross_norm_t(y_t),          # [B, 64, 1024]  y_t 전용 norm
            key=self.cross_norm_s_key(y_s),        # [B, 50, 1024]  y_s 전용 norm
            value=self.cross_norm_s_key(y_s),      # [B, 50, 1024]  key와 동일 norm
        )
        y_t = y_t + y_t_enriched            # [B, 64, 1024]  residual

        # NOTE: TSM을 붙일 때는 여기(y_t 바로 아래)에 삽입
        # y_t = self.tsm(y_t)

        # ── 5. Head ───────────────────────────────────────────────────────
        if detect:
            # y_t는 이미 [B, T=64, C] — adaptive_avg_pool1d 불필요
            # y_s를 global하게 요약해 temporal axis에 broadcast
            y_s_global = y_s.amax(dim=1, keepdim=True).expand(-1, y_t.size(1), -1)
            # [B, 1, C] → [B, 64, C]
            y_i = torch.cat([y_t, y_s_global], dim=-1)  # [B, 64, 2048]
            return self.fc(y_i)                          # [B, 64, num_class]

        y_t_pool = y_t.amax(dim=1)                      # [B, 1024]
        y_s_pool = y_s.amax(dim=1)                      # [B, 1024]
        y_i = torch.cat([y_t_pool, y_s_pool], dim=-1)   # [B, 2048]
        if knn_eval:
            return y_i
        return self.fc(y_i)                              # [B, num_class]