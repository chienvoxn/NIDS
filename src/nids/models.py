"""Gaussian anomaly and CNN/SE/temporal attention components."""
import numpy as np
import torch
from torch import nn

class DiagonalGaussian:
    def __init__(self, eps=1e-6):
        self.eps = eps
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mu_ = X.mean(axis=0)
        self.var_ = np.maximum(X.var(axis=0), self.eps)
        return self
    def score_samples(self, X):
        x = np.asarray(X, dtype=np.float64)
        return -0.5 * np.sum(np.log(2*np.pi*self.var_) + (x-self.mu_)**2/self.var_, axis=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                 nn.Linear(channels, max(1, channels // reduction)), nn.ReLU(),
                                 nn.Linear(max(1, channels // reduction), channels), nn.Sigmoid())
    def forward(self, x):
        return x * self.net(x).unsqueeze(-1)

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(channels, channels//2), nn.Tanh(), nn.Linear(channels//2, 1))
    def forward(self, x, mask=None):
        # x: batch, time, channels. mask: True = observed.
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(logits, dim=1)
        if mask is not None:
            weights = weights * mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (x * weights.unsqueeze(-1)).sum(dim=1), weights

class SEATCNN(nn.Module):
    """Only meaningful for genuine time ordered flow sequences [B,T,F]."""
    def __init__(self, features, classes, use_se=True, use_attention=True, hidden=64):
        super().__init__()
        self.use_attention = use_attention
        self.conv = nn.Sequential(nn.Conv1d(features, hidden, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU())
        self.se = SEBlock(hidden) if use_se else nn.Identity()
        self.attention = TemporalAttention(hidden) if use_attention else None
        self.classifier = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, classes))
    def forward(self, x, mask=None):
        if x.ndim != 3:
            raise ValueError('Expected [batch,time,features], not tabular [batch,features]')
        z = self.se(self.conv(x.transpose(1,2))).transpose(1,2)
        if self.attention:
            pooled, _ = self.attention(z, mask)
        elif mask is None:
            pooled = z.mean(dim=1)
        else:
            m = mask.float().unsqueeze(-1)
            pooled = (z*m).sum(1) / m.sum(1).clamp_min(1)
        return self.classifier(pooled)
