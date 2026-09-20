# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0

"""Earlier single-Gaussian model. Only here so we can score checkpoints/dbian_best.pt."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBED_COLS, EMBED_DIM

# checkpoint was trained with 32-d embeddings + 8 continuous columns = 40
LEGACY_INPUT_DIM = 40
LEGACY_CONT_DIM = 8

# don't change these if you still want to load checkpoints/dbian_best.pt
CROSS_LAYERS = 4
N_HEADS = 4
ATTN_DK = 16
ATTN_DV = LEGACY_INPUT_DIM
FUSION_DIM = LEGACY_INPUT_DIM * 2
MLP_HIDDEN = [128, 64, 32]
DROPOUT = 0.3


class EntityEmbedding(nn.Module):
    def __init__(self, cardinalities: Dict[str, int], embed_dim: int = 8):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(n, embed_dim) for col, n in cardinalities.items()
        })

    def forward(self, cat_indices: torch.Tensor) -> torch.Tensor:
        parts = [self.embeddings[col](cat_indices[:, i]) for i, col in enumerate(EMBED_COLS)]
        return torch.cat(parts, dim=-1)


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 4):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x_l = x0
        for w, b in zip(self.weights, self.biases):
            x_l = x0 * F.linear(x_l, w) + b + x_l
        return x_l


class FeatureSelfAttention(nn.Module):
    def __init__(self, n_features: int, n_heads: int = 4, d_k: int = 16, d_v: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v or n_features
        self.q_proj = nn.Linear(1, n_heads * d_k)
        self.k_proj = nn.Linear(1, n_heads * d_k)
        self.v_proj = nn.Linear(1, n_heads * self.d_v)
        self.out_proj = nn.Linear(n_heads * self.d_v, 1)
        self.ffn = nn.Sequential(
            nn.Linear(n_features, n_features * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(n_features * 2, n_features),
        )
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, d = z.shape
        x = z.unsqueeze(-1)
        q = self.q_proj(x).view(b, d, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(b, d, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(b, d, self.n_heads, self.d_v).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(b, d, -1)
        z_att = self.norm1(z + self.out_proj(context).squeeze(-1))
        return self.norm2(z_att + self.ffn(z_att))


class DBIAN(nn.Module):
    def __init__(self, cardinalities: Dict[str, int]):
        super().__init__()
        self.entity_emb = EntityEmbedding(cardinalities, EMBED_DIM)
        self.cross_net = CrossNetwork(LEGACY_INPUT_DIM, CROSS_LAYERS)
        self.feature_attn = FeatureSelfAttention(LEGACY_INPUT_DIM, N_HEADS, ATTN_DK, ATTN_DV, dropout=DROPOUT)
        layers: List[nn.Module] = []
        in_dim = FUSION_DIM
        for hidden in MLP_HIDDEN:
            layers.extend([nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(DROPOUT)])
            in_dim = hidden
        self.mlp = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, 1)
        self.nu_head = nn.Linear(in_dim, 1)

    def forward(self, cat: torch.Tensor, cont: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([self.entity_emb(cat), cont[:, :LEGACY_CONT_DIM]], dim=-1)
        fused = torch.cat([self.cross_net(z), self.feature_attn(z)], dim=-1)
        h = self.mlp(fused)
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.nu_head(h).squeeze(-1)) + 1e-4
        return mu, sigma

    @torch.no_grad()
    def predict(self, cat: torch.Tensor, cont: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
        self.train()
        preds = [self.forward(cat, cont)[0] for _ in range(n_samples)]
        self.eval()
        return torch.stack(preds, dim=0).mean(dim=0)
