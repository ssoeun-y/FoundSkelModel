import torch, math, warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Skeleton_Emb(nn.Module):
    def __init__(self, t_input_size, s_input_size, hidden_size) -> None:
        super().__init__()
        self.t_embedding = nn.Sequential(
            nn.Linear(t_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.s_embedding = nn.Sequential(
            nn.Linear(s_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t_src, s_src):
        t_src = self.t_embedding(t_src)
        s_src = self.s_embedding(s_src)
        return t_src, s_src


class CausalTemporalLinear(nn.Module):
    """
    Strictly online (causal) version of Linear(seqlen, seqlen).

    Key idea:
      - weight shape = [seqlen, seqlen]  →  SAME as pretrained, can load directly
      - forward: mask weight with lower-triangular matrix before applying
        → position t can only attend to positions 0..t (no future leakage)

    This satisfies all three conditions:
      1. Strictly causal (online): frame t sees only frames 0..t        ✅
      2. Same shape as original Linear(seqlen,seqlen): pretrained loads ✅
      3. No future leakage anywhere in DSA                              ✅
    """
    def __init__(self, seqlen):
        super().__init__()
        self.seqlen = seqlen
        # Same parameter names as nn.Linear so pretrained weights load directly
        self.weight = nn.Parameter(torch.zeros(seqlen, seqlen))
        self.bias   = nn.Parameter(torch.zeros(seqlen))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        # lower-triangular mask: tril[i,j]=1 if j<=i, else 0
        mask = torch.tril(torch.ones(seqlen, seqlen))
        self.register_buffer('mask', mask)

    def forward(self, x):
        # x: [N, C, T]  (Linear applied on last dim = T)
        T = x.size(-1)
        # apply causal mask: zero out upper triangle (future positions)
        w = self.weight[:T, :T] * self.mask[:T, :T]
        return F.linear(x, w, self.bias[:T])


class DSA(nn.Module):
    """
    Dense Shift Attention — strictly online (causal) version.

    wt1, wt2 use CausalTemporalLinear:
      - Same weight shape as original Linear(seqlen,seqlen) → pretrained compatible
      - Lower-triangular mask in forward → frame t cannot see t+1, t+2, ...
    Attention also uses causal_mask → no future leakage in attention either.
    """
    def __init__(self, seqlen, dim, gamma, gap, attn):
        super().__init__()
        self.gamma = gamma
        self.gap = gap
        # ✅ Same shape as pretrained Linear(seqlen,seqlen), but causal in forward
        self.wt1 = CausalTemporalLinear(seqlen)
        self.wt2 = CausalTemporalLinear(seqlen)
        self.act  = nn.ReLU()
        self.drop = DropPath(0.1)
        self.attn = attn
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp  = MLP(dim, dim * 2, dim)

    def forward(self, x):
        _, MT, C = x.shape          # [N, T, C]
        F1 = x.permute(0, 2, 1)    # [N, C, T]

        # Dense Shift: causal linear — frame t sees only 0..t
        F_h = self.wt2(self.act(self.wt1(F1))) + F1   # [N, C, T]

        # restore anchor frames (gap positions keep original)
        indices = torch.arange(0, MT, self.gap, device=F_h.device)
        F_h[:, :, indices] = F1[:, :, indices]

        F_h = F_h.permute(0, 2, 1)   # [N, T, C]

        # causal attention mask: position t cannot attend to t+1, t+2, ...
        causal_mask = torch.triu(
            torch.ones(MT, MT, device=x.device), diagonal=1).bool()

        F_h_norm = self.norm1(F_h)
        Ftp1 = self.drop(
            self.attn(F_h_norm, F_h_norm, F_h_norm, attn_mask=causal_mask)[0]
        ) + F_h
        Ftp1 = Ftp1 + self.drop(self.mlp(self.norm2(Ftp1)))

        x_norm = self.norm1(x)
        Ftp2 = self.drop(
            self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)[0]
        ) + x
        Ftp2 = Ftp2 + self.drop(self.mlp(self.norm2(Ftp2)))

        return (Ftp1 + Ftp2) * 0.5


class CA(nn.Module):
    def __init__(self, dim, attn, kernel_size):
        super().__init__()
        self.attn  = attn
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel_size)
        self.act   = nn.ReLU()
        self.drop  = DropPath(0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, dim * 2, dim)

    def forward(self, x):
        N, MT, C = x.shape
        Fseq2 = x.permute(0, 2, 1)
        Ftc   = (Fseq2 + self.act(self.conv1(Fseq2))).permute(0, 2, 1)
        Ftc_norm = self.norm1(Ftc)
        causal_mask = torch.triu(
            torch.ones(MT, MT, device=x.device), diagonal=1).bool()
        Ftc = self.drop(
            self.attn(Ftc_norm, Ftc_norm, Ftc_norm, attn_mask=causal_mask)[0]
        ) + x
        return self.drop(self.mlp(self.norm2(Ftc)))


class DST_Layer(nn.Module):
    def __init__(self, seqlen, dim, alpha, beta, gap, attn, kernel_size):
        super().__init__()
        self.seqlen = seqlen
        self.CA  = CA(dim, attn, kernel_size)
        self.DSA = DSA(seqlen, dim, alpha, gap, attn)
        self.beta  = beta
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * self.CA(x) + self.beta * self.DSA(x)


class DSTE(nn.Module):
    def __init__(self, t_input_size, s_input_size,
                 hidden_size, num_head, num_layer, alpha, gap,
                 kernel_size=1) -> None:
        super().__init__()
        self.ske_emb = Skeleton_Emb(t_input_size, s_input_size, hidden_size)
        self.d_model = hidden_size
        self.tpe = PositionalEncoding(hidden_size)
        self.spe = torch.nn.Parameter(torch.zeros(1, 50, hidden_size))
        trunc_normal_(self.spe, std=.02)
        alpha, beta, gap = alpha, 1 - alpha, gap

        attn_t  = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=num_head,
                                        dropout=0., batch_first=True)
        attn_s  = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=num_head,
                                        dropout=0., batch_first=True)
        self.t_tr  = DST_Layer(seqlen=64, dim=hidden_size, alpha=alpha, beta=beta,
                               gap=gap, attn=attn_t, kernel_size=kernel_size)
        self.s_tr  = DST_Layer(seqlen=50, dim=hidden_size, alpha=alpha, beta=beta,
                               gap=gap, attn=attn_s, kernel_size=kernel_size)

        attn_t1 = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=num_head,
                                        dropout=0., batch_first=True)
        attn_s1 = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=num_head,
                                        dropout=0., batch_first=True)
        self.t_tr1 = DST_Layer(seqlen=64, dim=hidden_size, alpha=alpha, beta=beta,
                                gap=gap, attn=attn_t1, kernel_size=kernel_size)
        self.s_tr1 = DST_Layer(seqlen=50, dim=hidden_size, alpha=alpha, beta=beta,
                                gap=gap, attn=attn_s1, kernel_size=kernel_size)

    def forward(self, jt, js):
        t_src, s_src = self.ske_emb(jt, js)
        B, _, _ = t_src.shape
        y_t = self.t_tr(self.tpe(t_src))
        y_t = self.t_tr1(self.tpe(y_t))
        y_s = self.s_tr(s_src + self.spe.expand(B, -1, -1))
        y_s = self.s_tr1(y_s + self.spe.expand(B, -1, -1))
        return y_t, y_s


class Downstream(nn.Module):
    def __init__(self, t_input_size, s_input_size,
                 hidden_size, num_head, num_layer, num_class=60,
                 modality='joint', alpha=0.5, gap=4, kernel_size=1) -> None:
        super().__init__()
        self.modality = modality
        self.d_model  = 2 * hidden_size
        self.backbone = DSTE(t_input_size, s_input_size, hidden_size,
                             num_head, num_layer,
                             alpha=alpha, gap=gap, kernel_size=kernel_size)
        self.fc = nn.Linear(self.d_model, num_class)

    def forward(self, jt, js, bt, bs, mt, ms, knn_eval=False, detect=False):
        if self.modality == 'joint':
            y_t, y_s = self.backbone(jt, js)
        elif self.modality == 'motion':
            y_t, y_s = self.backbone(mt, ms)
        elif self.modality == 'bone':
            y_t, y_s = self.backbone(bt, bs)

        if detect:
            y_s = F.adaptive_avg_pool1d(
                y_s.permute(0, 2, 1), 64).permute(0, 2, 1)
            y_i = torch.cat([y_t, y_s], dim=-1)
            return self.fc(y_i)

        y_t, y_s = y_t.amax(dim=1), y_s.amax(dim=1)
        y_i = torch.cat([y_t, y_s], dim=-1)
        if knn_eval:
            return y_i
        return self.fc(y_i)


# strictly online causal aliases
DSTECausal       = DSTE
DownstreamCausal = Downstream
