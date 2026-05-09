"""
AFCM.py — Anticipatory Future Context Module
─────────────────────────────────────────────
입력:  y_t  [B, T, H]   (causal backbone의 temporal feature)
출력:  fused [B, T, H]  (shape 완전 동일, 기존 head 그대로 사용 가능)

동작:
  - 각 시점 t에서 K={2,4,6} 프레임 뒤의 feature를 예측
  - 예측된 future features를 weighted sum으로 fusion
  - 학습 시: L_future = MSE(predicted, actual future features)
  - 추론 시: predicted future만 사용 → 완전 causal 유지

Causal 보장:
  - predictor 내부도 causal attention mask 사용
  - future feature는 예측값이지 실제값이 아님
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Lightweight causal self-attention (single head, 기존 backbone과 일관성)"""
    def __init__(self, hidden_size: int, num_head: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_head,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()
        out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        return self.norm(x + self.dropout(out))


class FuturePredictor(nn.Module):
    """
    K 프레임 뒤의 feature를 예측하는 lightweight causal transformer block.
    입력/출력 shape: [B, T, H] → [B, T, H]
    위치 t의 출력 = t+K 위치의 feature 예측값
    """
    def __init__(self, hidden_size: int, num_head: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attn_block = CausalSelfAttention(hidden_size, num_head, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        x = self.attn_block(x)
        x = self.norm(x + self.ffn(x))
        return x  # [B, T, H]


class AFCM(nn.Module):
    """
    Anticipatory Future Context Module

    K_steps = [2, 4, 6]: 각 K에 대해 별도 predictor 사용
    fusion: learned weighted sum of predicted futures + original

    학습 시 loss 계산:
      loss = afcm(y_t, compute_loss=True) → fused, L_future
    추론 시:
      fused = afcm(y_t, compute_loss=False)
    """
    def __init__(
        self,
        hidden_size: int,
        num_head:    int   = 1,
        k_steps:     list  = [2, 4, 6],
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.k_steps     = k_steps

        # K별 별도 predictor
        self.predictors = nn.ModuleList([
            FuturePredictor(hidden_size, num_head, dropout)
            for _ in k_steps
        ])

        # fusion weight: original + K개 predicted futures
        n_sources = 1 + len(k_steps)
        self.fusion_weight = nn.Parameter(
            torch.ones(n_sources) / n_sources
        )

        # output projection (shape 유지)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm     = nn.LayerNorm(hidden_size)

    def forward(
        self,
        y_t:          torch.Tensor,          # [B, T, H]
        compute_loss: bool = False,
    ):
        """
        Args:
            y_t:          [B, T, H] causal backbone temporal output
            compute_loss: True이면 (fused, L_future) 반환
                          False이면 fused만 반환

        Returns:
            fused:    [B, T, H]  (shape 동일, 기존 head 그대로 사용)
            L_future: scalar tensor (compute_loss=True일 때만)
        """
        B, T, H = y_t.shape

        predicted_futures = []
        for predictor in self.predictors:
            pred = predictor(y_t)   # [B, T, H]
            predicted_futures.append(pred)

        # Fusion: softmax로 normalized weight
        w = F.softmax(self.fusion_weight, dim=0)   # (1 + K,)

        fused = w[0] * y_t
        for i, pred in enumerate(predicted_futures):
            fused = fused + w[i + 1] * pred        # [B, T, H]

        fused = self.norm(self.out_proj(fused))    # [B, T, H]

        if not compute_loss:
            return fused

        # ── Future prediction loss (학습 시만) ──────────────────────────
        # predicted_futures[i][t] ≈ y_t[t + k_steps[i]]
        # 마지막 k_steps[i] 프레임은 GT가 없으므로 제외
        L_future = torch.tensor(0.0, device=y_t.device)
        count = 0
        with torch.no_grad():
            target = y_t.detach()   # actual features as supervision

        for i, k in enumerate(self.k_steps):
            if T - k <= 0:
                continue
            # 위치 0..T-k-1에서의 예측 vs 실제 t+k feature
            pred_slice   = predicted_futures[i][:, :T - k, :]   # [B, T-k, H]
            target_slice = target[:, k:, :]                      # [B, T-k, H]
            L_future = L_future + F.mse_loss(pred_slice, target_slice)
            count += 1

        if count > 0:
            L_future = L_future / count

        return fused, L_future
