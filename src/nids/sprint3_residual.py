"""Multi-scale residual CNN with optional SE and MHSA over feature tokens.

This treats tabular feature *order* as a modeling choice, not true time order.
No reliance on original V10 checkpoint; requires fresh training.
"""
from __future__ import annotations
import torch
from torch import nn


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, width) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, width))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.pre = nn.LayerNorm(width)
        self.branches = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=k, padding=k // 2)
            for k in (1, 3, 5)
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(3 * width, width, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.post = nn.LayerNorm(width)

    def forward(self, x):
        # x: [batch, feature_tokens, width]
        y = self.pre(x).transpose(1, 2)
        y = self.fuse(torch.cat([conv(y) for conv in self.branches], dim=1))
        y = y.transpose(1, 2)
        return self.post(x + y if self.residual else y)


class SEBlock(nn.Module):
    def __init__(self, width: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, width // reduction)
        self.gate = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                  nn.Linear(hidden, width), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x.mean(dim=1)).unsqueeze(1)


class MultiScaleResidualCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int, width: int = 32,
                 heads: int = 4, blocks: int = 2, dropout: float = 0.1,
                 residual: bool = True, se: bool = True, mhsa: bool = True):
        super().__init__()
        if width % heads != 0:
            raise ValueError('width must be divisible by heads')
        if blocks < 1 or n_features < 1 or n_classes < 2:
            raise ValueError('Invalid model dimensions')
        self.tokenizer = FeatureTokenizer(n_features, width)
        self.convs = nn.Sequential(*[
            MultiScaleResidualBlock(width, dropout, residual=residual)
            for _ in range(blocks)
        ])
        self.se = SEBlock(width) if se else nn.Identity()
        self.mhsa = nn.MultiheadAttention(width, heads, dropout=dropout,
                                          batch_first=True) if mhsa else None
        self.attn_norm = nn.LayerNorm(width) if mhsa else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.Linear(width, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(width * 3),
                                        nn.Linear(width * 3, width * 2),
                                        nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(width * 2, n_classes))

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.tokenizer.weight.shape[0]:
            raise ValueError('Expected [batch, n_features]')
        z = self.convs(self.tokenizer(x))
        z = self.se(z)
        if self.mhsa is not None:
            q = self.attn_norm(z)
            mixed, _ = self.mhsa(q, q, q, need_weights=False)
            z = z + self.dropout(mixed)
        weights = torch.softmax(self.pool(z).squeeze(-1), dim=1)
        rep = torch.cat([(z * weights.unsqueeze(-1)).sum(dim=1),
                         z.mean(dim=1), z.amax(dim=1)], dim=1)
        return self.classifier(rep)


ABLATIONS = {
    'ms_cnn': {'residual': False, 'se': False, 'mhsa': False},
    'ms_residual': {'residual': True, 'se': False, 'mhsa': False},
    'ms_residual_se': {'residual': True, 'se': True, 'mhsa': False},
    'ms_residual_se_mhsa': {'residual': True, 'se': True, 'mhsa': True},
}
